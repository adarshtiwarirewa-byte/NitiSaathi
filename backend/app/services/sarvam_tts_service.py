"""Sarvam Bulbul text-to-speech adapter."""

import base64
import os

import httpx
from dotenv import load_dotenv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

API_URL = "https://api.sarvam.ai/text-to-speech"
MODEL_NAME = os.getenv("SARVAM_TTS_MODEL", "bulbul:v4-flash")
SUPPORTED_V4_LANGUAGES = {
    "en-IN", "hi-IN", "bn-IN", "gu-IN", "kn-IN",
    "mr-IN", "pa-IN", "ta-IN", "te-IN",
}


class SarvamTTSUnavailableError(RuntimeError):
    """Raised when Sarvam TTS cannot complete a speech request."""


def generate_speech(text: str, language_code: str, speaker: str) -> bytes:
    """Synthesize exact answer text as WAV using Bulbul v4 Flash."""
    if not text or not text.strip():
        raise ValueError("Speech text cannot be empty.")
    if len(text) > 2500:
        raise ValueError("Bulbul v4 supports up to 2,500 characters per request.")
    if language_code not in SUPPORTED_V4_LANGUAGES:
        raise ValueError("This language is not currently enabled for Bulbul v4.")

    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise SarvamTTSUnavailableError(
            "SARVAM_API_KEY is not configured. Add it to the backend environment."
        )

    try:
        response = httpx.post(
            API_URL,
            headers={"api-subscription-key": api_key},
            json={
                "text": text.strip(),
                "language_code": language_code,
                "model": MODEL_NAME,
                "speaker": speaker,
                "output_audio_codec": "wav",
                "speech_sample_rate": 24000,
                "pace": 1.0,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        audio_items = response.json().get("audios", [])
        if not audio_items:
            raise SarvamTTSUnavailableError("Sarvam returned no audio.")
        return base64.b64decode(audio_items[0], validate=True)
    except SarvamTTSUnavailableError:
        raise
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        raise SarvamTTSUnavailableError(
            "Sarvam voice generation failed. Check the API key, voice, and selected language, then retry."
        ) from exc
