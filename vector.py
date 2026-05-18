import os
import pickle
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# ─── Configuration ─────────────────────────────────────────────────────────────

PDF_PATH = "IPCC_AR6_SYR_FullVolume.pdf"
CHROMA_PATH = "chroma_db"
CHUNKS_CACHE = "chunks_cache.pkl"  # cache pour ne pas recouper le PDF à chaque run

# mxbai-embed-large a une fenetre de 512 tokens (~2 000 chars).
# On reste a 800 chars pour eviter toute troncature tout en preservant
# suffisamment de contexte autour des chiffres.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# Pages du Résumé pour Décideurs (SPM) dans l'AR6 SYR FullVolume.
# Vérification empirique : la page 55 (1-indexé) = 54 (0-indexé) marque
# la fin du SPM. PyPDFLoader numérote à partir de 0.
SPM_MAX_PAGE = 54

embeddings = OllamaEmbeddings(model="mxbai-embed-large")


# ─── Chargement & découpage ────────────────────────────────────────────────────


def _load_and_split() -> List[Document]:
    """
    Charge le PDF, découpe en chunks et enrichit les métadonnées.
    Le résultat est mis en cache sur disque pour accélérer les runs suivants.
    """
    if os.path.exists(CHUNKS_CACHE):
        print("[INFO] Chargement des chunks depuis le cache disque...")
        with open(CHUNKS_CACHE, "rb") as f:
            return pickle.load(f)

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Le fichier {PDF_PATH} est introuvable.")

    print("[INFO] Chargement et découpage du PDF (première fois, ~quelques minutes)...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Les séparateurs sont ordonnés du plus coarse au plus fin :
        # on évite de couper en plein milieu d'une phrase contenant un chiffre.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Enrichissement des métadonnées : "SPM" ou "corps"
    for chunk in chunks:
        page = chunk.metadata.get("page", 9999)
        chunk.metadata["section_type"] = "SPM" if page < SPM_MAX_PAGE else "corps"

    n_spm = sum(1 for c in chunks if c.metadata["section_type"] == "SPM")
    print(
        f"[INFO] {len(chunks)} fragments créés ({n_spm} SPM, {len(chunks)-n_spm} corps)."
    )

    with open(CHUNKS_CACHE, "wb") as f:
        pickle.dump(chunks, f)
    print("[INFO] Chunks mis en cache.")
    return chunks


# ─── Base vectorielle ──────────────────────────────────────────────────────────


def get_vector_store() -> Chroma:
    if os.path.exists(CHROMA_PATH):
        print("[INFO] Chargement de la base Chroma existante...")
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    chunks = _load_and_split()
    print("[INFO] Création de la base vectorielle Chroma...")
    db = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH
    )
    print("[INFO] Base vectorielle créée avec succès.")
    return db


# ─── Retrieval hybride ─────────────────────────────────────────────────────────


def _reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    weights: list[float],
    k_rrf: int = 60,
) -> list[Document]:
    """
    Reciprocal Rank Fusion (RRF) — fusionne N listes de résultats classés
    sans dépendance externe.

    Score RRF d'un document d dans la liste i :
        score(d) += weight_i / (k_rrf + rank_i(d))

    k_rrf = 60 est la valeur standard (Cormack et al., 2009) ; elle atténue
    l'impact des documents très bien classés dans une seule liste.
    Les listes sont pondérées : dense 0.6 / BM25 0.4 pour favoriser
    la sémantique tout en gardant la précision lexicale sur les chiffres.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(ranked_list, start=1):
            key = doc.page_content[:120]
            scores[key] = scores.get(key, 0.0) + weight / (k_rrf + rank)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys]


def _hybrid_search(query: str, k: int) -> list[Document]:
    """
    Combine deux signaux complémentaires via RRF :
    - Dense (sémantique) : capture les paraphrases et synonymes
    - BM25  (lexical)    : capture les termes exacts, unités, chiffres précis
    """
    db = get_vector_store()
    chunks = _load_and_split()

    dense_results = db.similarity_search(query, k=k)
    bm25_results = BM25Retriever.from_documents(chunks, k=k).invoke(query)

    return _reciprocal_rank_fusion(
        ranked_lists=[dense_results, bm25_results],
        weights=[0.6, 0.4],
    )


def _deduplicate(docs: List[Document]) -> List[Document]:
    """Supprime les doublons en comparant les 120 premiers caractères de chaque chunk."""
    seen, unique = set(), []
    for doc in docs:
        key = doc.page_content[:120]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


# ─── Point d'entrée public ─────────────────────────────────────────────────────


def get_retrieved_context(query: str, k: int = 6) -> str:
    """
    Stratégie de retrieval à deux niveaux :

    Niveau 1 — Retrieval hybride (dense + BM25) sur la requête.
    Niveau 2 — Injection de chunks SPM supplémentaires si le niveau 1
               n'en a pas remonté assez (garantit la présence des
               conclusions globales du rapport dans le contexte).

    Chaque chunk est préfixé de sa page et de sa section pour permettre
    au LLM de hiérarchiser les sources.
    """
    results = _deduplicate(_hybrid_search(query, k))

    # ── Injection de sécurité : toujours au moins 2 chunks SPM ──────────────
    spm_count = sum(1 for d in results if d.metadata.get("section_type") == "SPM")

    if spm_count < 2:
        db = get_vector_store()
        extra_spm = db.similarity_search(query, k=3, filter={"section_type": "SPM"})
        existing_keys = {d.page_content[:120] for d in results}
        for doc in extra_spm:
            if doc.page_content[:120] not in existing_keys:
                results.append(doc)
                existing_keys.add(doc.page_content[:120])

    # ── Formatage avec métadonnées sources ───────────────────────────────────
    parts = []
    for doc in results[:k]:
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section_type", "corps")
        label = f"[Page {page} — {section.upper()}]"
        parts.append(f"{label}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)
