
from pathlib import Path
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer


# Project storage location
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

# Multilingual embedding model
MODEL_NAME = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load the embedding model once and reuse it.
    """

    return SentenceTransformer(MODEL_NAME)


def get_collection():
    """
    Connect to the persistent ChromaDB database.
    """

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name="niti_saathi_documents",
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def index_chunks(chunks: list[dict]) -> dict:
    """
    Convert document chunks into embeddings
    and store them in ChromaDB.
    """

    if not chunks:
        return {
            "success": True,
            "indexed_chunks": 0
        }

    model = get_embedding_model()
    collection = get_collection()

    # Prepare text for multilingual E5
    texts = [
        "passage: " + chunk["text"]
        for chunk in chunks
    ]

    # Generate numerical embeddings
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Prepare metadata
    metadatas = [
        {
            "document_id": chunk["document_id"],
            "filename": chunk["filename"],
            "page_number": chunk["page_number"],
            "word_count": chunk["word_count"]
        }
        for chunk in chunks
    ]

    # Store embeddings and original text
    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    return {
        "success": True,
        "indexed_chunks": len(chunks),
        "document_id": chunks[0]["document_id"]
    }
