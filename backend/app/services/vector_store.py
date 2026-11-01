
from pathlib import Path
from functools import lru_cache



PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

MODEL_NAME = "intfloat/multilingual-e5-small"

LANGUAGE_METADATA_FIELDS = (
    "language_label",
    "language_code",
    "script",
    "detection_model",
    "detection_raw_score",
    "latin_ratio",
)


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load the embedding model once and reuse it.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def get_collection():
    """
    Connect to the persistent ChromaDB database.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    import chromadb

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
    and store them in ChromaDB with language metadata.
    """

    if not chunks:
        return {
            "success": True,
            "indexed_chunks": 0
        }

    model = get_embedding_model()
    collection = get_collection()

    # Prepare text for multilingual E5.
    texts = [
        "passage: " + chunk["text"]
        for chunk in chunks
    ]

    # Generate numerical embeddings.
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Prepare metadata.
    metadatas = []

    for chunk in chunks:
        metadata = {
            "document_id": chunk["document_id"],
            "filename": chunk["filename"],
            "page_number": chunk["page_number"],
            "word_count": chunk["word_count"],
        }

        # Add language information when available.
        # Optional fields allow older chunks and existing tests
        # to continue working.
        for field in LANGUAGE_METADATA_FIELDS:
            value = chunk.get(field)

            if value is not None:
                metadata[field] = value

        metadatas.append(metadata)

    # Store embeddings, original text and metadata.
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
