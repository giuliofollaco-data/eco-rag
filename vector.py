import os
import pickle
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# ── Configuration ───────────────────────────────────────────────────────────────

PDF_PATH = "IPCC_AR6_SYR_FullVolume.pdf"
CHROMA_PATH = "chroma_db"
CHUNKS_CACHE = "chunks_cache.pkl"

# Chunks de 800 caracteres avec 200 caracteres de recouvrement pour garder du contexte entre les fragments.
# Ce choix est empirique : il permet de limiter le nombre de chunks entrée de mxbai-embed-large (2048 tokens) tout en conservant une cohérence suffisante dans les fragments.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# Pages du Resume pour Decideurs (SPM) dans l'AR6 SYR FullVolume.
# Verification empirique : la page 55 (1-indexe) = 54 (0-indexe) marque
# la fin du SPM. PyPDFLoader numerote a partir de 0.
SPM_MAX_PAGE = 54

embeddings = OllamaEmbeddings(model="mxbai-embed-large")


# ── Chargement & decoupage ──────────────────────────────────────────────────────


def _load_and_split() -> List[Document]:
    """
    Charge le PDF, decoupe en chunks et enrichit les metadonnees.
    Le resultat est mis en cache sur disque pour accelerer les runs suivants.
    """
    if os.path.exists(CHUNKS_CACHE):
        print("[INFO] Chargement des chunks depuis le cache disque...")
        with open(CHUNKS_CACHE, "rb") as f:
            return pickle.load(f)

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Le fichier {PDF_PATH} est introuvable.")

    print("[INFO] Chargement et decoupage du PDF (premiere fois, ~quelques minutes)...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Enrichissement des metadonnees : "SPM" ou "corps"
    for chunk in chunks:
        page = chunk.metadata.get("page", 9999)
        chunk.metadata["section_type"] = "SPM" if page < SPM_MAX_PAGE else "corps"

    n_spm = sum(1 for c in chunks if c.metadata["section_type"] == "SPM")
    print(
        f"[INFO] {len(chunks)} fragments crees ({n_spm} SPM, {len(chunks)-n_spm} corps)."
    )

    with open(CHUNKS_CACHE, "wb") as f:
        pickle.dump(chunks, f)
    print("[INFO] Chunks mis en cache.")
    return chunks


# ── Base vectorielle ────────────────────────────────────────────────────────────


def get_vector_store() -> Chroma:
    if os.path.exists(CHROMA_PATH):
        print("[INFO] Chargement de la base Chroma existante...")
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    chunks = _load_and_split()
    print("[INFO] Creation de la base vectorielle Chroma...")
    db = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH
    )
    print("[INFO] Base vectorielle creee avec succes.")
    return db


# ── Retrieval hybride ───────────────────────────────────────────────────────────


def _build_ensemble_retriever(k: int) -> EnsembleRetriever:
    """
    Combine deux signaux complementaires :
    - Dense (semantique) : capture les paraphrases et synonymes
    - BM25  (lexical)    : capture les termes exacts, unites, chiffres precis

    Le poids BM25 de 0.4 est volontairement eleve pour les questions numeriques,
    ou la correspondance lexicale est plus fiable que la similarite vectorielle.
    """
    db = get_vector_store()
    chunks = _load_and_split()

    dense_retriever = db.as_retriever(search_kwargs={"k": k})
    bm25_retriever = BM25Retriever.from_documents(chunks, k=k)

    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever], weights=[0.6, 0.4]
    )


def _deduplicate(docs: List[Document]) -> List[Document]:
    """Supprime les doublons en comparant les 120 premiers caracteres de chaque chunk."""
    seen, unique = set(), []
    for doc in docs:
        key = doc.page_content[:120]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


# ── Point d'entree public ───────────────────────────────────────────────────────


def get_retrieved_context(query: str, k: int = 6) -> str:
    """
    Strategie de retrieval a deux niveaux :

    Niveau 1 — Retrieval hybride (dense + BM25) sur la requete.
    Niveau 2 — Injection de chunks SPM supplementaires si le niveau 1
               n'en a pas remonte assez (garantit la presence des
               conclusions globales du rapport dans le contexte).

    Chaque chunk est prefixe de sa page et de sa section pour permettre
    au LLM de hierarchiser les sources.
    """
    retriever = _build_ensemble_retriever(k)
    results = _deduplicate(retriever.invoke(query))

    # Injection de securite : toujours au moins 2 chunks SPM
    spm_count = sum(1 for d in results if d.metadata.get("section_type") == "SPM")

    if spm_count < 2:
        db = get_vector_store()
        extra_spm = db.similarity_search(query, k=3, filter={"section_type": "SPM"})
        existing_keys = {d.page_content[:120] for d in results}
        for doc in extra_spm:
            if doc.page_content[:120] not in existing_keys:
                results.append(doc)
                existing_keys.add(doc.page_content[:120])

    # Formatage avec metadonnees sources
    parts = []
    for doc in results[:k]:
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section_type", "corps")
        label = f"[Page {page} — {section.upper()}]"
        parts.append(f"{label}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)
