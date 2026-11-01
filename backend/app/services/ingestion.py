from pathlib import Path
from uuid import uuid4
import json

import fitz
from fastapi import HTTPException, UploadFile
from backend.app.services.chunking import create_chunks
from backend.app.services.vector_store import index_chunks
from backend.app.models.policy import PolicyDocument
from backend.app.services.indiclid_service import IndicLIDService

# Project storage locations
PROJECT_ROOT = Path(__file__).resolve().parents[3]

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


async def ingest_pdf(
    file: UploadFile,
    indiclid_service: IndicLIDService
) -> dict:
    """
    Validates a PDF, extracts text page by page,
    and stores the original PDF and extracted content.
    """

    # 1. Validate filename
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF documents are supported."
        )

    # 2. Read uploaded content with a size limit
    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the 20 MB upload limit."
        )

    # 3. Check PDF signature
    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF."
        )

    # 4. Extract text page by page
    try:
        with fitz.open(stream=content, filetype="pdf") as document:

            if document.needs_pass:
                raise HTTPException(
                    status_code=400,
                    detail="Password-protected PDFs are not supported."
                )

            pages = []

            for page_number, page in enumerate(document, start=1):

                text = page.get_text("text").strip()

                pages.append({
                    "page_number": page_number,
                    "text": text,
                    "character_count": len(text)
                })

    except HTTPException:
        raise

    except (fitz.FileDataError, RuntimeError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="Could not extract this PDF. Please check the document."
        )

    # 5. Reject scanned PDFs without extractable text
    if not any(page["text"] for page in pages):
        raise HTTPException(
            status_code=422,
            detail=(
                "No extractable text was found. "
                "This may be a scanned PDF. OCR is not supported yet."
            )
        )

    # 6. Generate a unique document ID
    document_id = str(uuid4())

    # 7. Create structured policy metadata
    policy_document = PolicyDocument(
        document_id=document_id,
        original_filename=file.filename,
        page_count=len(pages)
    )


    # 8. Create storage directories
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 9. Save original PDF
    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
    pdf_path.write_bytes(content)

    # 10. Save extracted text and page metadata
    extracted_data = {
        "document_id": document_id,
        "original_filename": file.filename,
        "page_count": len(pages),
        "policy_document": policy_document.model_dump(mode="json"),
        "pages": pages
    }

    json_path = PROCESSED_DIR / f"{document_id}.json"

    json_path.write_text(
        json.dumps(extracted_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


    # 11. Create searchable text chunks
    chunks = create_chunks(extracted_data)

    # 12. Detect the language of every chunk
    for chunk in chunks:
        language_result = indiclid_service.detect(chunk["text"])

        chunk.update(language_result)

    # 13. Save chunks separately
    chunks_path = PROCESSED_DIR / f"{document_id}_chunks.json"

    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    indexing_result = index_chunks(chunks)

    # 14. Return document information
    return {
        "document_id": document_id,
        "filename": file.filename,
        "page_count": len(pages),
        "total_characters": sum(
            page["character_count"] for page in pages
        ),
        "indexed_chunks": indexing_result["indexed_chunks"],
        "indexing_success": indexing_result["success"],

        "warnings": [
            "Some pages contain no extractable text."
        ] if any(not page["text"] for page in pages) else [],
        "page_previews": [
            {
                "page_number": page["page_number"],
                "preview": page["text"][:200],
                "character_count": page["character_count"]
            }
            for page in pages
        ],
        "policy_document": policy_document.model_dump(mode="json"),
    }