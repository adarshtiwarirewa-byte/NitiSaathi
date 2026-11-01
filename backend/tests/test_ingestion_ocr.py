import asyncio
import io
import json

import pymupdf as fitz
from fastapi import UploadFile

from backend.app.services import ingestion
from backend.app.services.ocr_service import OCRPageResult


class FakeIndicLIDService:
    def detect(self, text: str) -> dict:
        return {
            "language_label": "eng_Latn",
            "language_code": "eng",
            "script": "Latn",
            "detection_model": "test",
            "detection_raw_score": 0.99,
            "latin_ratio": 1.0,
        }


def create_mixed_pdf() -> bytes:
    document = fitz.open()

    first_page = document.new_page()
    first_page.insert_text(
        (72, 72),
        "This is a normal digital policy page with enough selectable text.",
        fontsize=11,
    )

    second_page = document.new_page()
    second_page.insert_text((72, 72), "Appendix", fontsize=11)
    second_page.draw_rect(
        fitz.Rect(60, 100, 500, 350),
        color=(0, 0, 0),
        width=1,
    )

    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_ingestion_automatically_selects_ocr_language_and_preserves_text(
    monkeypatch,
    tmp_path,
):
    pdf_bytes = create_mixed_pdf()

    monkeypatch.setattr(ingestion, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(ingestion, "PROCESSED_DIR", tmp_path / "processed")

    requested_pages = []

    def fake_extract_scanned_pages(pdf_bytes, page_indexes, language):
        requested_pages.extend(page_indexes)
        assert language == "auto"
        return {
            1: OCRPageResult(
                text="Appendix\nOCR extracted additional policy information.",
                languages_used=("hi-IN", "mr-IN", "en-IN"),
                detected_script="Devanagari",
                script_confidence=14.0,
                selection_method="tesseract_osd",
            )
        }

    monkeypatch.setattr(
        ingestion,
        "extract_scanned_pages",
        fake_extract_scanned_pages,
    )
    monkeypatch.setattr(
        ingestion,
        "index_chunks",
        lambda chunks: {"success": True, "indexed_chunks": len(chunks)},
    )

    async def run():
        upload = UploadFile(
            file=io.BytesIO(pdf_bytes),
            filename="mixed_policy.pdf",
        )
        return await ingestion.ingest_pdf(
            upload,
            FakeIndicLIDService(),
        )

    result = asyncio.run(run())

    assert requested_pages == [1]
    assert result["document_language_mode"] == "automatic"
    assert result["source_languages"] == ["eng"]
    assert result["ocr_pages"] == 1

    saved_json = (
        tmp_path
        / "processed"
        / f"{result['document_id']}.json"
    )
    extracted = json.loads(saved_json.read_text(encoding="utf-8"))

    first_page = extracted["pages"][0]
    second_page = extracted["pages"][1]

    assert first_page["extraction_method"] == "native"
    assert "normal digital policy page" in first_page["text"]

    assert second_page["extraction_method"] == "native_and_tesseract_ocr"
    assert "Appendix" in second_page["text"]
    assert "OCR extracted additional policy information" in second_page["text"]
    assert second_page["ocr_metadata"]["language_selection_method"] == "tesseract_osd"

    # Language is now recorded at text-segment and page level.
    assert second_page["page_languages"] == ["eng"]
    assert second_page["language_segments"]
    assert second_page["language_segments"][0]["language_code"] == "eng"
    assert result["page_previews"][1]["page_languages"] == ["eng"]

    assert extracted["policy_document"]["source_languages"] == ["eng"]

def test_page_with_native_text_and_embedded_image_is_candidate_for_partial_ocr():
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "This page already contains enough selectable policy text to pass the threshold.",
        fontsize=11,
    )

    image_document = fitz.open()
    image_page = image_document.new_page(width=80, height=40)
    image_page.draw_rect(
        fitz.Rect(5, 5, 75, 35),
        color=(0, 0, 0),
        fill=(1, 1, 1),
    )
    image_bytes = image_page.get_pixmap(alpha=False).tobytes("png")
    image_document.close()

    page.insert_image(
        fitz.Rect(72, 120, 152, 160),
        stream=image_bytes,
    )

    native_text = page.get_text("text").strip()
    assert len(native_text) >= ingestion.MIN_NATIVE_TEXT_CHARS
    assert ingestion._needs_ocr(page, native_text) is True

    document.close()