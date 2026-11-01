"""Segment-level language identification for multilingual policy pages."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


# A paragraph may contain several sentences; sentence boundaries let us detect
# language changes even when two languages use the same writing system.
_SEGMENT_BOUNDARY_RE = re.compile(r"(?<=[.!?।॥])\s+|\n[ \t]*\n+")
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_MIN_LANGUAGE_LETTERS = 24

_SCRIPT_MARKERS = (
    ("Devanagari", "DEVANAGARI"),
    ("Bengali", "BENGALI"),
    ("Gurmukhi", "GURMUKHI"),
    ("Gujarati", "GUJARATI"),
    ("Oriya", "ORIYA"),
    ("Tamil", "TAMIL"),
    ("Telugu", "TELUGU"),
    ("Kannada", "KANNADA"),
    ("Malayalam", "MALAYALAM"),
    ("Arabic", "ARABIC"),
    ("Latin", "LATIN"),
)


def _script_for_word(word: str) -> str | None:
    """Return the dominant Unicode script represented by an alphanumeric word."""
    counts: dict[str, int] = {}

    for character in word:
        if not character.isalpha():
            continue

        name = unicodedata.name(character, "")
        script = next(
            (label for label, marker in _SCRIPT_MARKERS if marker in name),
            "Other",
        )
        counts[script] = counts.get(script, 0) + 1

    if not counts:
        return None

    return max(counts, key=counts.get)


def _sentence_spans(text: str):
    """Yield trimmed sentence/paragraph spans with offsets in the original text."""
    start = 0

    for boundary in _SEGMENT_BOUNDARY_RE.finditer(text):
        raw_start, raw_end = start, boundary.start()
        while raw_start < raw_end and text[raw_start].isspace():
            raw_start += 1
        while raw_end > raw_start and text[raw_end - 1].isspace():
            raw_end -= 1

        if raw_start < raw_end:
            yield raw_start, raw_end, text[raw_start:raw_end]

        start = boundary.end()

    raw_start, raw_end = start, len(text)
    while raw_start < raw_end and text[raw_start].isspace():
        raw_start += 1
    while raw_end > raw_start and text[raw_end - 1].isspace():
        raw_end -= 1

    if raw_start < raw_end:
        yield raw_start, raw_end, text[raw_start:raw_end]


def _script_runs(text: str, base_offset: int):
    """Split a sentence into contiguous script runs while retaining offsets."""
    words = list(_WORD_RE.finditer(text))
    if not words:
        return

    current_script: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    for word_match in words:
        script = _script_for_word(word_match.group())
        if script is None:
            continue

        if current_script is None:
            current_script = script
            current_start = word_match.start()
            current_end = word_match.end()
            continue

        if script == current_script:
            current_end = word_match.end()
            continue

        assert current_start is not None and current_end is not None
        run_text = text[current_start:current_end].strip()
        leading = len(text[current_start:current_end]) - len(text[current_start:current_end].lstrip())
        trailing_text = text[current_start:current_end].rstrip()
        run_end = current_start + len(trailing_text)
        run_start = current_start + leading
        if run_text:
            yield (
                base_offset + run_start,
                base_offset + run_end,
                run_text,
                current_script,
            )

        current_script = script
        current_start = word_match.start()
        current_end = word_match.end()

    if current_script is not None and current_start is not None and current_end is not None:
        raw = text[current_start:current_end]
        run_text = raw.strip()
        leading = len(raw) - len(raw.lstrip())
        run_end = current_start + len(raw.rstrip())
        run_start = current_start + leading
        if run_text:
            yield (
                base_offset + run_start,
                base_offset + run_end,
                run_text,
                current_script,
            )


def analyze_page_languages(
    text: str,
    indiclid_service: Any,
    page_number: int,
) -> dict:
    """Detect languages across meaningful text segments on one page.

    A page may produce multiple language segments. Segment boundaries are
    sentence/paragraph boundaries plus changes in Unicode writing system.
    IndicLID is applied independently to each sufficiently long segment.
    Very short runs are retained as unclassified metadata rather than being
    force-labelled from a few characters.
    """
    segments: list[dict] = []
    page_languages: set[str] = set()
    page_scripts: set[str] = set()

    # Keep explicit offsets in a single pass, including repeated sentence text.
    segment_index = 0
    for sentence_start, _sentence_end, sentence_text in _sentence_spans(text):
        for start_char, end_char, segment_text, script_hint in _script_runs(
            sentence_text,
            sentence_start,
        ):
            segment_index += 1
            alphabetic_count = sum(char.isalpha() for char in segment_text)

            segment = {
                "segment_id": f"page-{page_number}-segment-{segment_index}",
                "start_char": start_char,
                "end_char": end_char,
                "text": segment_text,
                "script_hint": script_hint,
                "alphabetic_characters": alphabetic_count,
                "detection_status": "too_short",
                "language_label": None,
                "language_code": None,
                "script": None,
                "detection_model": None,
                "detection_raw_score": None,
            }

            if alphabetic_count >= _MIN_LANGUAGE_LETTERS:
                prediction = indiclid_service.detect(segment_text)
                segment.update({
                    "detection_status": "detected",
                    "language_label": prediction.get("language_label"),
                    "language_code": prediction.get("language_code"),
                    "script": prediction.get("script"),
                    "detection_model": prediction.get("detection_model"),
                    "detection_raw_score": prediction.get("detection_raw_score"),
                })

                if prediction.get("language_code"):
                    page_languages.add(prediction["language_code"])
                if prediction.get("script"):
                    page_scripts.add(prediction["script"])

            page_scripts.add(script_hint)
            segments.append(segment)

    return {
        "language_segments": segments,
        "page_languages": sorted(page_languages),
        "detected_scripts": sorted(page_scripts),
        "language_detection_method": "indic_lid_segment_analysis",
    }
