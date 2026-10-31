
from backend.app.services.chunking import create_chunks


def test_chunking():

    # Sample document using our existing JSON structure
    document = {
        "document_id": "test-document-001",
        "original_filename": "sample.pdf",
        "page_count": 1,
        "pages": [
            {
                "page_number": 1,
                "text": " ".join(
                    [f"word{i}" for i in range(1200)]
                ),
                "character_count": 0
            }
        ]
    }

    chunks = create_chunks(document)

    print("\nTotal chunks:", len(chunks))

    for chunk in chunks:
        print("\nChunk ID:", chunk["chunk_id"])
        print("Page:", chunk["page_number"])
        print("Word count:", chunk["word_count"])
        print("Preview:", chunk["text"][:100])

    assert len(chunks) == 3
    assert chunks[0]["word_count"] == 500
    assert chunks[1]["word_count"] == 500
    assert chunks[2]["word_count"] == 300

    assert chunks[0]["document_id"] == "test-document-001"

    print("\nAll chunking tests passed!")


if __name__ == "__main__":
    test_chunking()
