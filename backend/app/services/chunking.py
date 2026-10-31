
import re
from uuid import uuid4


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def clean_text(text: str) -> str:
    """
    Remove unnecessary whitespace from extracted text.
    Preserve the actual words and Unicode characters.
    """

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_chunks(document: dict) -> list[dict]:
    """
    Split extracted document text into smaller chunks.

    Each chunk retains its document ID, filename,
    page number and unique chunk ID.
    """

    chunks = []

    document_id = document["document_id"]
    filename = document["original_filename"]

    for page in document["pages"]:

        page_number = page["page_number"]
        text = clean_text(page["text"])

        if not text:
            continue

        words = text.split()

        start = 0

        while start < len(words):

            end = start + CHUNK_SIZE

            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_id": str(uuid4()),
                "document_id": document_id,
                "filename": filename,
                "page_number": page_number,
                "text": chunk_text,
                "word_count": len(chunk_words)
            })

            if end >= len(words):
                break

            start = end - CHUNK_OVERLAP

    return chunks
