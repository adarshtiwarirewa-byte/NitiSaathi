
from backend.app.services.vector_store import (
    index_chunks,
    get_collection
)


def test_vector_store():

    # Create sample chunks
    chunks = [
        {
            "chunk_id": "test-chunk-001",
            "document_id": "test-document-001",
            "filename": "sample.pdf",
            "page_number": 1,
            "text": "Farmers receive financial assistance from the government.",
            "word_count": 8
        },
        {
            "chunk_id": "test-chunk-002",
            "document_id": "test-document-001",
            "filename": "sample.pdf",
            "page_number": 2,
            "text": "Eligible farmers can apply for government financial benefits.",
            "word_count": 9
        }
    ]

    # Generate embeddings and store chunks
    result = index_chunks(chunks)

    print("\nIndexing result:", result)

    # Verify database storage
    collection = get_collection()

    print("Total stored chunks:", collection.count())

    stored_data = collection.get(
        ids=["test-chunk-001", "test-chunk-002"],
        include=["documents", "metadatas"]
    )

    print("\nStored documents:")

    for document, metadata in zip(
        stored_data["documents"],
        stored_data["metadatas"]
    ):
        print("\nText:", document)
        print("Metadata:", metadata)

    # Assertions
    assert result["success"] is True
    assert result["indexed_chunks"] == 2
    assert len(stored_data["documents"]) == 2

    print("\nAll vector store tests passed!")


if __name__ == "__main__":
    test_vector_store()
