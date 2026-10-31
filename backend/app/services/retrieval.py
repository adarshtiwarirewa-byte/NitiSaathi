
from backend.app.services.vector_store import (
    get_embedding_model,
    get_collection
)


def search_documents(
    question: str,
    top_k: int = 3
) -> list[dict]:
    """
    Search for the most relevant document chunks
    using multilingual semantic similarity.
    """

    if not question or not question.strip():
        return []

    model = get_embedding_model()
    collection = get_collection()

    # Convert user question into an embedding
    query_embedding = model.encode(
        ["query: " + question.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Retrieve similar document chunks
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []

    if not results["documents"] or not results["documents"][0]:
        return []

    MAX_DISTANCE = 0.25

    for i, text in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        if distance > MAX_DISTANCE:
            continue

        retrieved_chunks.append({
            "text": text,
            "document_id": metadata["document_id"],
            "filename": metadata["filename"],
            "page_number": metadata["page_number"],
            "distance": distance
        })

    return retrieved_chunks
