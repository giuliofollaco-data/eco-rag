import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Paramètres
PDF_PATH = "IPCC_AR6_SYR_FullVolume.pdf"
CHROMA_PATH = "chroma_db"  # Dossier où sera sauvegardée la base de données

# 1. Initialisation du modèle d'embedding (doit tourner localement via Ollama)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")


def get_vector_store():
    """Charge la base vectorielle si elle existe, sinon la crée à partir du PDF."""
    if os.path.exists(CHROMA_PATH):
        print("[INFO] Chargement de la base de données vectorielle Chroma existante...")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    else:
        print(
            "[INFO] Création de la base de données... (Cela peut prendre plusieurs minutes)"
        )

        # 2. Charger le document PDF
        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"Le fichier {PDF_PATH} est introuvable.")

        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()

        # 3. Découper le texte en morceaux (chunks)
        # On utilise un recouvrement (overlap) pour ne pas couper le contexte entre deux paragraphes
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"[INFO] Document découpé en {len(chunks)} fragments.")

        # 4. Créer et persister la base vectorielle Chroma
        db = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH
        )
        print("[INFO] Base vectorielle créée avec succès.")

    return db


def get_retrieved_context(query: str, k: int = 4) -> str:
    """Recherche les 'k' fragments les plus pertinents pour une question donnée."""
    db = get_vector_store()

    # Recherche de similarité
    results = db.similarity_search(query, k=k)

    # Extraction et fusion du texte des résultats
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    return context_text
