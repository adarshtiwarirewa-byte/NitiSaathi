"""Sarvam Bulbul text-to-speech adapter with language-aware model routing."""

import base64
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

API_URL = "https://api.sarvam.ai/text-to-speech"

# Keep the currently tested v4 voices for Hindi and Indian English.
# Use Bulbul v3 for the additional regional languages because Sarvam's
# documented language-specific speaker recommendations use v3 speaker IDs.
BULBUL_V4_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v4-flash")
BULBUL_V3_MODEL = os.getenv("SARVAM_TTS_V3_MODEL", "bulbul:v3")

TTS_VOICE_CONFIG = {
    "en-IN": {
        "model": BULBUL_V4_MODEL,
        "speaker": "simran_en_customer",
    },
    "hi-IN": {
        "model": BULBUL_V4_MODEL,
        "speaker": "aparna_hi_customer",
    },
    "bn-IN": {
        "model": BULBUL_V3_MODEL,
        "speaker": "roopa",
    },
    "gu-IN": {
        "model": BULBUL_V3_MODEL,
        "speaker": "priya",
    },
    "kn-IN": {
        "model": BULBUL_V3_MODEL,
        "speaker": "ishita",
    },
    "mr-IN": {
        "model": BULBUL_V3_MODEL,
        "speaker": "priya",
    },
    "pa-IN": {
        "model": BULBUL_V3_MODEL,
        "speaker": "roopa",
    },
    "ta-IN": {
        "model": BULBUL_V3_MODEL,
        "speaker": "ishita",
    },
    "te-IN": {
        "model": BULBUL_V3_MODEL,
        "speaker": "priya",
    },
}


class SarvamTTSUnavailableError(RuntimeError):
    """Raised when Sarvam TTS cannot complete a speech request."""


def generate_speech(text: str, language_code: str, speaker: str) -> bytes:
    """Synthesize text as WAV using the configured language-specific voice."""

    if not text or not text.strip():
        raise ValueError("Speech text cannot be empty.")

    clean_text = text.strip()

    if len(clean_text) > 2500:
        raise ValueError(
            "Sarvam Bulbul supports up to 2,500 characters per request."
        )

    voice_config = TTS_VOICE_CONFIG.get(language_code)

    if voice_config is None:
        supported = ", ".join(TTS_VOICE_CONFIG)
        raise ValueError(
            f"This language is not enabled for NitiSaathi speech. Supported: {supported}."
        )

    expected_speaker = voice_config["speaker"]

    if speaker != expected_speaker:
        raise ValueError(
            f"Invalid speaker for {language_code}. "
            f"Use the configured NitiSaathi voice '{expected_speaker}'."
        )

    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key:
        raise SarvamTTSUnavailableError(
            "SARVAM_API_KEY is not configured. Add it to the backend environment."
        )

    payload = {
        "text": clean_text,
        "language_code": language_code,
        "model": voice_config["model"],
        "speaker": expected_speaker,
        "output_audio_codec": "wav",
        "speech_sample_rate": 24000,
        "pace": 1.0,
    }

    try:
        response = httpx.post(
            API_URL,
            headers={"api-subscription-key": api_key},
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()

        audio_items = response.json().get("audios", [])
        if not audio_items:
            raise SarvamTTSUnavailableError("Sarvam returned no audio.")

        return base64.b64decode(audio_items[0], validate=True)

    except SarvamTTSUnavailableError:
        raise
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise SarvamTTSUnavailableError(
            "Sarvam voice generation failed. Check the API key, voice, and selected language, then retry."
        ) from exc
