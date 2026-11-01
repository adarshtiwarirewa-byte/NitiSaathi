from __future__ import annotations

from pathlib import Path
from uuid import uuid4
from typing import TYPE_CHECKING
import json

import pymupdf as fitz
from fastapi import HTTPException, UploadFile

from backend.app.services.chunking import create_chunks
from backend.app.services.vector_store import index_chunks
from backend.app.models.policy import PolicyDocument
if TYPE_CHECKING:
    from backend.app.services.indiclid_service import IndicLIDService
from backend.app.services.language_analysis import analyze_page_languages
from backend.app.services.ocr_service import (
    LocalOCRError,
    extract_scanned_pages,
    TESSERACT_LANGUAGES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
MIN_NATIVE_TEXT_CHARS = 30


def _needs_ocr(page: fitz.Page, native_text: str) -> bool:
    """Identify pages with missing, sparse, or visibly broken text layers."""

    if "\ufffd" in native_text:
        return True

    # Images may contain tables, stamps, or additional policy text even when
    # the same page already has a selectable text layer. Partial OCR preserves
    # that native text and examines only regions without legible text.
    if page.get_images():
        return True

    drawings = page.get_drawings()

    if len(native_text) < MIN_NATIVE_TEXT_CHARS:
        return bool(drawings)

    # A large number of vector drawing objects can indicate text converted
    # into outlines. Avoid OCR for ordinary pages with just a few table lines.
    return len(drawings) >= 80


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


async def ingest_pdf(
    file: UploadFile,
    indiclid_service: IndicLIDService,
    document_language: str = "auto",
) -> dict:
    """Extract, OCR when needed, identify languages, and index a PDF."""

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF documents are supported.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the 20 MB upload limit.",
        )

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF.",
        )

    if document_language.strip().lower() != "auto":
        requested_languages = [
            item.strip()
            for item in document_language.split("+")
        ]
        if (
            not requested_languages
            or any(
                language_code not in TESSERACT_LANGUAGES
                for language_code in requested_languages
            )
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported document language for OCR. "
                    "Use 'auto' or a supported Indian language/English code."
                ),
            )

    # Extract native text and identify pages that need OCR.
    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            if document.needs_pass:
                raise HTTPException(
                    status_code=400,
                    detail="Password-protected PDFs are not supported.",
                )

            pages = []
            for page_index, page in enumerate(document):
                native_text = page.get_text("text").strip()
                pages.append({
                    "page_number": page_index + 1,
                    "text": native_text,
                    "character_count": len(native_text),
                    "extraction_method": (
                        "pending_ocr"
                        if _needs_ocr(page, native_text)
                        else "native"
                    ),
                })

    except HTTPException:
        raise
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail="Could not extract this PDF. Please check the document.",
        ) from exc

    scanned_page_indexes = [
        index
        for index, page in enumerate(pages)
        if page["extraction_method"] == "pending_ocr"
    ]

    if scanned_page_indexes:
        try:
            ocr_pages = extract_scanned_pages(
                content,
                page_indexes=scanned_page_indexes,
                language=document_language,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LocalOCRError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        for page_index in scanned_page_indexes:
            ocr_result = ocr_pages.get(page_index)
            if ocr_result is None:
                if pages[page_index]["text"]:
                    pages[page_index]["extraction_method"] = "native"
                else:
                    pages[page_index]["extraction_method"] = "unreadable"
                continue

            original_text = pages[page_index]["text"]
            ocr_text = ocr_result.text.strip()

            pages[page_index]["ocr_metadata"] = {
                "languages_used": list(ocr_result.languages_used),
                "detected_script": ocr_result.detected_script,
                "script_confidence": ocr_result.script_confidence,
                "language_selection_method": ocr_result.selection_method,
            }

            if not ocr_text:
                pages[page_index]["extraction_method"] = (
                    "native" if original_text else "unreadable"
                )
                continue

            pages[page_index]["text"] = ocr_text
            pages[page_index]["character_count"] = len(ocr_text)

            if not original_text:
                pages[page_index]["extraction_method"] = "tesseract_ocr"
            elif _normalize_whitespace(original_text) == _normalize_whitespace(ocr_text):
                pages[page_index]["extraction_method"] = "native"
            else:
                pages[page_index]["extraction_method"] = "native_and_tesseract_ocr"

    if not any(page["text"] for page in pages):
        raise HTTPException(
            status_code=422,
            detail=(
                "No readable text was found in this PDF. "
                "Check the scan quality and document orientation."
            ),
        )

    # Identify every language-bearing segment on each page. A single page
    # may contain several languages and/or writing systems.
    for page in pages:
        page.update(
            analyze_page_languages(
                page["text"],
                indiclid_service,
                page_number=page["page_number"],
            )
        )

    document_id = str(uuid4())

    extracted_data = {
        "document_id": document_id,
        "original_filename": file.filename,
        "page_count": len(pages),
        "pages": pages,
    }

    # Keep the existing page-sized retrieval chunks, and continue to detect
    # their language as a second signal. Segment-level predictions above make
    # mixed-language pages visible rather than assigning one label per page.
    chunks = create_chunks(extracted_data)
    for chunk in chunks:
        chunk.update(indiclid_service.detect(chunk["text"]))

    chunk_languages_by_page: dict[int, set[str]] = {}
    for chunk in chunks:
        language_code = chunk.get("language_code")
        if language_code:
            chunk_languages_by_page.setdefault(
                chunk["page_number"], set()
            ).add(language_code)

    document_languages: set[str] = set()
    for page in pages:
        segment_languages = set(page.get("page_languages", []))
        chunk_languages = chunk_languages_by_page.get(
            page["page_number"], set()
        )
        page_languages = segment_languages | chunk_languages

        page["chunk_languages"] = sorted(chunk_languages)
        page["page_languages"] = sorted(page_languages)
        page["language_detection_method"] = (
            "segment_and_chunk_indiclid"
            if segment_languages and chunk_languages
            else "segment_indiclid"
            if segment_languages
            else "chunk_fallback"
            if chunk_languages
            else "undetected"
        )
        document_languages.update(page_languages)

    source_languages = sorted(document_languages)

    policy_document = PolicyDocument(
        document_id=document_id,
        original_filename=file.filename,
        page_count=len(pages),
        source_languages=source_languages,
    )

    extracted_data["policy_document"] = policy_document.model_dump(mode="json")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
    pdf_path.write_bytes(content)

    json_path = PROCESSED_DIR / f"{document_id}.json"
    json_path.write_text(
        json.dumps(extracted_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    chunks_path = PROCESSED_DIR / f"{document_id}_chunks.json"
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    indexing_result = index_chunks(chunks)

    successful_ocr_methods = {
        "tesseract_ocr",
        "native_and_tesseract_ocr",
    }
    ocr_page_count = sum(
        page["extraction_method"] in successful_ocr_methods
        for page in pages
    )

    unreadable_pages = [
        page["page_number"] for page in pages if not page["text"]
    ]
    warnings = []

    if unreadable_pages:
        page_numbers = ", ".join(str(number) for number in unreadable_pages)
        warnings.append(
            f"Text could not be read from page(s): {page_numbers}. "
            "Those pages were not indexed."
        )

    fallback_pages = [
        page["page_number"]
        for page in pages
        if page.get("ocr_metadata", {}).get("language_selection_method")
        == "fallback_all_languages"
    ]
    if fallback_pages:
        page_numbers = ", ".join(str(number) for number in fallback_pages)
        warnings.append(
            f"Automatic script detection was uncertain on page(s): {page_numbers}. "
            "OCR tried all supported language packs on those pages."
        )

    page_previews = []
    for page in pages:
        preview = {
            "page_number": page["page_number"],
            "preview": page["text"][:200],
            "character_count": page["character_count"],
            "extraction_method": page["extraction_method"],
            "page_languages": page.get("page_languages", []),
            "detected_scripts": page.get("detected_scripts", []),
            "language_segment_count": len(page.get("language_segments", [])),
        }
        if "ocr_metadata" in page:
            preview["ocr_metadata"] = page["ocr_metadata"]
        page_previews.append(preview)

    return {
        "document_id": document_id,
        "filename": file.filename,
        "page_count": len(pages),
        "total_characters": sum(page["character_count"] for page in pages),
        "indexed_chunks": indexing_result["indexed_chunks"],
        "indexing_success": indexing_result["success"],
        "ocr_pages": ocr_page_count,
        "source_languages": source_languages,
        "document_language_mode": (
            "automatic" if document_language.strip().lower() == "auto" else "manual_override"
        ),
        "warnings": warnings,
        "page_previews": page_previews,
        "policy_document": policy_document.model_dump(mode="json"),
    }
