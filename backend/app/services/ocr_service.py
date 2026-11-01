"""Local Tesseract OCR adapter with automatic script-based language selection."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess

import pymupdf as fitz
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


# Public upload/API language codes mapped to Tesseract language packs.
TESSERACT_LANGUAGES = {
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

ALL_SUPPORTED_OCR_LANGUAGES = tuple(TESSERACT_LANGUAGES)
OSD_SCRIPT_CONFIDENCE_THRESHOLD = 5.0
OSD_RENDER_DPI = 200

# Tesseract OSD identifies scripts, not exact languages. Pair the detected
# script with appropriate OCR packs. English is included on Indic pages
# because government PDFs are commonly bilingual.
SCRIPT_LANGUAGE_CANDIDATES = {
    "Latin": ("en-IN",),
    "Devanagari": ("hi-IN", "mr-IN", "en-IN"),
    "Bengali": ("bn-IN", "en-IN"),
    "Gujarati": ("gu-IN", "en-IN"),
    "Gurmukhi": ("pa-IN", "en-IN"),
    "Kannada": ("kn-IN", "en-IN"),
    "Tamil": ("ta-IN", "en-IN"),
    "Telugu": ("te-IN", "en-IN"),
}


class LocalOCRError(RuntimeError):
    """Raised when local OCR cannot process one or more pages."""


@dataclass(frozen=True)
class OCRPageResult:
    """OCR text plus information about how its OCR language was selected."""

    text: str
    languages_used: tuple[str, ...]
    detected_script: str | None
    script_confidence: float | None
    selection_method: str


def _find_tesseract_executable() -> str | None:
    """Find Tesseract using configuration, PATH, and common Windows paths."""

    configured_path = os.getenv("TESSERACT_PATH")

    if configured_path:
        configured = Path(configured_path).expanduser()

        if configured.is_dir():
            for name in ("tesseract.exe", "tesseract"):
                candidate = configured / name
                if candidate.is_file():
                    return str(candidate)
        elif configured.is_file():
            return str(configured)

        resolved = shutil.which(configured_path)
        if resolved:
            return resolved

    executable = shutil.which("tesseract")
    if executable:
        return executable

    if os.name == "nt":
        known_locations = (
            Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
            Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
        )
        for candidate in known_locations:
            if candidate.is_file():
                return str(candidate)

    return None


def _candidate_tessdata_dirs(executable: str | None) -> list[Path]:
    """Return likely tessdata directories without assuming one install layout."""

    candidates: list[Path] = [PROJECT_ROOT / "data" / "ocr" / "tessdata"]

    configured_prefix = os.getenv("TESSDATA_PREFIX")
    if configured_prefix:
        configured = Path(configured_prefix).expanduser()
        candidates.extend((configured, configured / "tessdata"))

    if executable:
        candidates.append(Path(executable).parent / "tessdata")

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _find_tessdata_dir(
    required_packs: list[str],
    executable: str | None,
) -> str | None:
    """Find a tessdata folder containing all requested traineddata files."""

    for candidate in _candidate_tessdata_dirs(executable):
        if candidate.is_dir() and all(
            (candidate / f"{pack}.traineddata").is_file()
            for pack in required_packs
        ):
            return str(candidate)

    return None


def _add_executable_directory_to_path(executable: str) -> None:
    """Make Tesseract discoverable by PyMuPDF for this process."""

    executable_directory = str(Path(executable).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)

    if executable_directory not in path_entries:
        os.environ["PATH"] = (
            executable_directory
            + os.pathsep
            + os.environ.get("PATH", "")
        )


def _parse_osd_output(output: str) -> tuple[str | None, float | None]:
    """Extract the detected script and its confidence from Tesseract OSD."""

    script_match = re.search(
        r"^\s*Script:\s*(.+?)\s*$",
        output,
        flags=re.MULTILINE,
    )
    confidence_match = re.search(
        r"^\s*Script confidence:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
        output,
        flags=re.MULTILINE,
    )

    if not script_match or not confidence_match:
        return None, None

    try:
        confidence = float(confidence_match.group(1))
    except ValueError:
        return None, None

    return script_match.group(1).strip(), confidence


def _detect_page_script(
    page: fitz.Page,
    executable: str,
    tessdata_dir: str,
) -> tuple[str | None, float | None]:
    """Use Tesseract OSD on a rendered page to identify its dominant script."""

    pixmap = page.get_pixmap(dpi=OSD_RENDER_DPI, alpha=False)
    image_bytes = pixmap.tobytes("png")

    command = [
        executable,
        "stdin",
        "stdout",
        "--tessdata-dir",
        tessdata_dir,
        "-l",
        "osd",
        "--psm",
        "0",
    ]

    try:
        completed = subprocess.run(
            command,
            input=image_bytes,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None

    output = (
        completed.stdout.decode("utf-8", errors="replace")
        + "\n"
        + completed.stderr.decode("utf-8", errors="replace")
    )

    if completed.returncode != 0:
        return None, None

    return _parse_osd_output(output)


def _parse_manual_languages(document_language: str) -> tuple[str, ...]:
    """Validate and normalize explicit OCR language overrides."""

    requested = tuple(
        item.strip()
        for item in document_language.split("+")
    )
    if not requested or any(
        item not in TESSERACT_LANGUAGES for item in requested
    ):
        raise ValueError(
            f"Unsupported OCR language '{document_language}'. "
            "Choose 'auto' or a supported Indian language/English code."
        )
    return requested


def _resolve_languages(
    page: fitz.Page,
    document_language: str,
    executable: str,
) -> tuple[tuple[str, ...], str | None, float | None, str]:
    """Resolve OCR packs from an explicit override or automatic OSD."""

    if document_language.strip().lower() != "auto":
        requested = _parse_manual_languages(document_language)
        return requested, None, None, "manual_override"

    osd_tessdata = _find_tessdata_dir(["osd"], executable)
    if not osd_tessdata:
        return ALL_SUPPORTED_OCR_LANGUAGES, None, None, "fallback_all_languages"

    script, confidence = _detect_page_script(
        page,
        executable,
        osd_tessdata,
    )

    if (
        script
        and confidence is not None
        and confidence >= OSD_SCRIPT_CONFIDENCE_THRESHOLD
        and script in SCRIPT_LANGUAGE_CANDIDATES
    ):
        return (
            SCRIPT_LANGUAGE_CANDIDATES[script],
            script,
            confidence,
            "tesseract_osd",
        )

    # OSD can be uncertain on sparse, noisy, or low-resolution pages.
    # Fall back to the installed supported language packs rather than
    # asking the user to identify the source language.
    return (
        ALL_SUPPORTED_OCR_LANGUAGES,
        script,
        confidence,
        "fallback_all_languages",
    )


def extract_scanned_pages(
    pdf_bytes: bytes,
    page_indexes: list[int],
    language: str = "auto",
) -> dict[int, OCRPageResult]:
    """OCR selected pages and report each page's automatic language selection."""

    if not page_indexes:
        return {}

    if not isinstance(language, str) or not language.strip():
        raise ValueError("OCR language must be 'auto' or a supported language code.")

    # Validate explicit overrides before trying to open the PDF or locate OCR.
    # This keeps input-validation errors deterministic and easy to test.
    if language.strip().lower() != "auto":
        _parse_manual_languages(language)

    executable = _find_tesseract_executable()
    if not executable:
        raise LocalOCRError(
            "Tesseract OCR was not found. Install Tesseract and configure "
            "TESSERACT_PATH or add it to PATH."
        )

    _add_executable_directory_to_path(executable)
    extracted: dict[int, OCRPageResult] = {}

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            for page_index in page_indexes:
                if (
                    not isinstance(page_index, int)
                    or isinstance(page_index, bool)
                    or page_index < 0
                    or page_index >= document.page_count
                ):
                    raise ValueError("OCR page index is outside the PDF.")

                page = document[page_index]
                languages, script, confidence, selection_method = (
                    _resolve_languages(page, language, executable)
                )

                tesseract_packs = [
                    TESSERACT_LANGUAGES[item]
                    for item in languages
                ]
                tesseract_language = "+".join(tesseract_packs)
                tessdata = _find_tessdata_dir(
                    tesseract_packs,
                    executable,
                )

                ocr_options = {
                    "language": tesseract_language,
                    "dpi": 300,
                    "full": False,
                }
                if tessdata:
                    ocr_options["tessdata"] = tessdata

                text_page = page.get_textpage_ocr(**ocr_options)
                page_text = page.get_text(
                    "text",
                    textpage=text_page,
                    sort=True,
                ).strip()

                extracted[page_index] = OCRPageResult(
                    text=page_text,
                    languages_used=languages,
                    detected_script=script,
                    script_confidence=confidence,
                    selection_method=selection_method,
                )

    except ValueError:
        raise
    except (fitz.FileDataError, RuntimeError, OSError) as exc:
        raise LocalOCRError(
            "Local OCR could not process this page. Check the Tesseract "
            "installation and the selected language data, then retry."
        ) from exc

    return extracted
