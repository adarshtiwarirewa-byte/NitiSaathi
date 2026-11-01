import pytest
import pymupdf as fitz

from backend.app.services import ocr_service
from backend.app.services.ocr_service import (
    LocalOCRError,
    OCRPageResult,
    TESSERACT_LANGUAGES,
    _parse_osd_output,
    extract_scanned_pages,
)


def create_test_pdf(page_count: int = 1) -> bytes:
    document = fitz.open()
    for _ in range(page_count):
        document.new_page()
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_supported_source_languages_map_to_tesseract_packs():
    assert TESSERACT_LANGUAGES == {
        "en-IN": "eng",
        "hi-IN": "hin",
        "bn-IN": "ben",
        "ta-IN": "tam",
        "te-IN": "tel",
        "mr-IN": "mar",
        "gu-IN": "guj",
        "kn-IN": "kan",
        "pa-IN": "pan",
    }


def test_unsupported_manual_language_is_rejected():
    with pytest.raises(ValueError, match="Unsupported OCR language"):
        extract_scanned_pages(b"", [0], "English")


def test_unsupported_member_of_combined_language_is_rejected():
    with pytest.raises(ValueError, match="Unsupported OCR language"):
        extract_scanned_pages(b"", [0], "en-IN+xx-IN")


def test_empty_page_indexes_return_empty_result():
    assert extract_scanned_pages(b"", [], "auto") == {}


def test_osd_output_is_parsed():
    output = (
        "Page number: 0\n"
        "Orientation in degrees: 0\n"
        "Rotate: 0\n"
        "Orientation confidence: 12.34\n"
        "Script: Devanagari\n"
        "Script confidence: 8.50\n"
    )
    assert _parse_osd_output(output) == ("Devanagari", 8.5)


def test_bad_osd_output_returns_unknown():
    assert _parse_osd_output("OSD could not detect script") == (None, None)


def test_auto_osd_selects_script_specific_language_packs(monkeypatch):
    calls = []
    fake_text_page = object()

    monkeypatch.setattr(
        ocr_service,
        "_find_tesseract_executable",
        lambda: "C:/fake/tesseract.exe",
    )
    monkeypatch.setattr(
        ocr_service,
        "_find_tessdata_dir",
        lambda required, executable: (
            "C:/fake/tessdata" if required == ["osd"] else None
        ),
    )
    monkeypatch.setattr(
        ocr_service,
        "_detect_page_script",
        lambda page, executable, tessdata: ("Devanagari", 14.0),
    )

    def fake_get_textpage_ocr(self, **kwargs):
        calls.append(kwargs)
        return fake_text_page

    def fake_get_text(self, option="text", *, textpage=None, sort=False, **kwargs):
        assert option == "text"
        assert textpage is fake_text_page
        assert sort is True
        return "यह एक परीक्षण दस्तावेज़ है. English content."

    monkeypatch.setattr(fitz.Page, "get_textpage_ocr", fake_get_textpage_ocr)
    monkeypatch.setattr(fitz.Page, "get_text", fake_get_text)

    result = extract_scanned_pages(
        create_test_pdf(),
        page_indexes=[0],
        language="auto",
    )

    assert isinstance(result[0], OCRPageResult)
    assert result[0].languages_used == ("hi-IN", "mr-IN", "en-IN")
    assert result[0].detected_script == "Devanagari"
    assert result[0].selection_method == "tesseract_osd"
    assert calls[0]["language"] == "hin+mar+eng"
    assert calls[0]["full"] is False
    assert calls[0]["dpi"] == 300


def test_low_osd_confidence_uses_automatic_fallback(monkeypatch):
    calls = []
    fake_text_page = object()

    monkeypatch.setattr(
        ocr_service,
        "_find_tesseract_executable",
        lambda: "C:/fake/tesseract.exe",
    )
    monkeypatch.setattr(
        ocr_service,
        "_find_tessdata_dir",
        lambda required, executable: (
            "C:/fake/tessdata" if required == ["osd"] else None
        ),
    )
    monkeypatch.setattr(
        ocr_service,
        "_detect_page_script",
        lambda page, executable, tessdata: ("Tamil", 1.0),
    )

    def fake_get_textpage_ocr(self, **kwargs):
        calls.append(kwargs)
        return fake_text_page

    monkeypatch.setattr(fitz.Page, "get_textpage_ocr", fake_get_textpage_ocr)
    monkeypatch.setattr(
        fitz.Page,
        "get_text",
        lambda self, *args, **kwargs: "fallback OCR text",
    )

    result = extract_scanned_pages(create_test_pdf(), [0], "auto")

    assert result[0].selection_method == "fallback_all_languages"
    assert len(result[0].languages_used) == len(TESSERACT_LANGUAGES)
    assert calls[0]["language"] == "eng+hin+ben+tam+tel+mar+guj+kan+pan"


def test_manual_language_override_skips_osd(monkeypatch):
    calls = []
    fake_text_page = object()

    monkeypatch.setattr(
        ocr_service,
        "_find_tesseract_executable",
        lambda: "C:/fake/tesseract.exe",
    )
    monkeypatch.setattr(
        ocr_service,
        "_find_tessdata_dir",
        lambda required, executable: None,
    )

    def osd_must_not_run(*args, **kwargs):
        raise AssertionError("OSD should not run for a manual override")

    monkeypatch.setattr(ocr_service, "_detect_page_script", osd_must_not_run)

    def fake_get_textpage_ocr(self, **kwargs):
        calls.append(kwargs)
        return fake_text_page

    monkeypatch.setattr(fitz.Page, "get_textpage_ocr", fake_get_textpage_ocr)
    monkeypatch.setattr(
        fitz.Page,
        "get_text",
        lambda self, *args, **kwargs: "manual language OCR text",
    )

    result = extract_scanned_pages(create_test_pdf(), [0], "ta-IN+en-IN")

    assert result[0].languages_used == ("ta-IN", "en-IN")
    assert result[0].selection_method == "manual_override"
    assert calls[0]["language"] == "tam+eng"


def test_missing_tesseract_is_actionable(monkeypatch):
    monkeypatch.setattr(
        ocr_service,
        "_find_tesseract_executable",
        lambda: None,
    )

    with pytest.raises(LocalOCRError, match="Tesseract OCR was not found"):
        extract_scanned_pages(create_test_pdf(), [0], "auto")


def test_invalid_page_index_is_rejected(monkeypatch):
    monkeypatch.setattr(
        ocr_service,
        "_find_tesseract_executable",
        lambda: "C:/fake/tesseract.exe",
    )

    with pytest.raises(ValueError, match="page index is outside"):
        extract_scanned_pages(create_test_pdf(), [4], "auto")
