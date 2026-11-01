"""Run a real, isolated Tesseract OCR test with automatic language selection."""

import asyncio
import io
import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf as fitz
from fastapi import UploadFile

from backend.app.services import ingestion


class TestIndicLID:
    def detect(self, text: str) -> dict:
        return {
            "language_label": "eng_Latn",
            "language_code": "eng",
            "script": "Latn",
            "detection_model": "smoke-test",
            "detection_raw_score": 0.99,
            "latin_ratio": 1.0,
        }


def create_scanned_pdf() -> bytes:
    """Render English text to a raster image and place it in an image-only PDF."""

    source = fitz.open()
    source_page = source.new_page(width=595, height=842)
    source_page.insert_text(
        (40, 100),
        "NitiSaathi OCR Verification",
        fontsize=24,
    )
    source_page.insert_text(
        (40, 160),
        "Farmers may receive financial assistance.",
        fontsize=18,
    )

    pixmap = source_page.get_pixmap(
        matrix=fitz.Matrix(3, 3),
        alpha=False,
    )
    image_bytes = pixmap.tobytes("png")
    source.close()

    scanned = fitz.open()
    page = scanned.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=image_bytes)
    pdf_bytes = scanned.tobytes()
    scanned.close()
    return pdf_bytes


async def main():
    pdf_bytes = create_scanned_pdf()

    original_upload_dir = ingestion.UPLOAD_DIR
    original_processed_dir = ingestion.PROCESSED_DIR
    original_index_chunks = ingestion.index_chunks
    uploaded_file = None

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ingestion.UPLOAD_DIR = temp_path / "uploads"
            ingestion.PROCESSED_DIR = temp_path / "processed"
            ingestion.index_chunks = lambda chunks: {
                "success": True,
                "indexed_chunks": len(chunks),
            }

            uploaded_file = UploadFile(
                file=io.BytesIO(pdf_bytes),
                filename="ocr-auto-language-test.pdf",
            )

            result = await ingestion.ingest_pdf(
                file=uploaded_file,
                indiclid_service=TestIndicLID(),
                # Omit document_language deliberately: automatic mode is default.
            )

            json_path = (
                ingestion.PROCESSED_DIR
                / f"{result['document_id']}.json"
            )
            extracted = json.loads(json_path.read_text(encoding="utf-8"))
            page = extracted["pages"][0]
            extracted_text = page["text"].lower()

            print("\nAUTOMATIC OCR LANGUAGE TEST")
            print("---------------------------")
            print("Detected source languages:", result["source_languages"])
            print("OCR pages:", result["ocr_pages"])
            print("OCR metadata:", page.get("ocr_metadata"))
            print("Extraction method:", page["extraction_method"])
            print("Extracted text:", page["text"])

            assert result["ocr_pages"] == 1
            assert "financial" in extracted_text
            assert "assistance" in extracted_text
            assert page["ocr_metadata"]["languages_used"]

            print("\nREAL TESSERACT AUTO-LANGUAGE OCR: PASSED")

    finally:
        if uploaded_file is not None:
            await uploaded_file.close()
        ingestion.UPLOAD_DIR = original_upload_dir
        ingestion.PROCESSED_DIR = original_processed_dir
        ingestion.index_chunks = original_index_chunks


if __name__ == "__main__":
    asyncio.run(main())
