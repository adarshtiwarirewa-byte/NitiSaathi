"""Provider router for NitiSaathi's structured explanation generation."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from pathlib import Path

from backend.app.models.explanation import PolicyExplanationDraft
from backend.app.services.gemini_service import generate_answer as generate_gemini_answer
from backend.app.services.sarvam_llm_service import (
    SarvamLLMUnavailableError,
    generate_sarvam_answer,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


def _fallback_enabled() -> bool:
    return os.getenv("LLM_FALLBACK_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    language: str = "English",
) -> PolicyExplanationDraft:
    """Use the configured LLM provider; optionally fall back to Gemini.

    Defaults to Sarvam-105B. Gemini remains available as a fallback and can
    also be selected directly with LLM_PROVIDER=gemini.
    """

    provider = os.getenv("LLM_PROVIDER", "sarvam").strip().lower()

    if provider == "gemini":
        return generate_gemini_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
            language=language,
        )

    if provider != "sarvam":
        raise ValueError(
            "Unsupported LLM_PROVIDER. Choose 'sarvam' or 'gemini'."
        )

    try:
        return generate_sarvam_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
            language=language,
        )
    except SarvamLLMUnavailableError as sarvam_error:
        if not _fallback_enabled():
            raise

        if not os.getenv("GEMINI_API_KEY"):
            raise SarvamLLMUnavailableError(
                f"{sarvam_error} Gemini fallback is enabled but GEMINI_API_KEY is not configured."
            ) from sarvam_error

        logger.warning(
            "Sarvam LLM failed; falling back to Gemini. Reason: %s",
            sarvam_error,
        )

        return generate_gemini_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
            language=language,
        )
