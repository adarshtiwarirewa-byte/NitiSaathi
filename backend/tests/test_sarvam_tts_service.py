import base64

import pytest

from backend.app.services import sarvam_tts_service as service


EXPECTED_CONFIG = {
    "en-IN": ("simran_en_customer", "bulbul:v4-flash"),
    "hi-IN": ("aparna_hi_customer", "bulbul:v4-flash"),
    "bn-IN": ("roopa", "bulbul:v3"),
    "gu-IN": ("priya", "bulbul:v3"),
    "kn-IN": ("ishita", "bulbul:v3"),
    "mr-IN": ("priya", "bulbul:v3"),
    "pa-IN": ("roopa", "bulbul:v3"),
    "ta-IN": ("ishita", "bulbul:v3"),
    "te-IN": ("priya", "bulbul:v3"),
}


@pytest.mark.parametrize("language_code,expected", EXPECTED_CONFIG.items())
def test_language_has_model_specific_voice(language_code, expected):
    config = service.TTS_VOICE_CONFIG[language_code]
    assert config["speaker"] == expected[0]
    assert config["model"] == expected[1]


def test_unsupported_language_is_rejected():
    with pytest.raises(ValueError, match="not enabled"):
        service.generate_speech("Hello", "ml-IN", "pooja")


def test_wrong_speaker_for_language_is_rejected():
    with pytest.raises(ValueError, match="Invalid speaker"):
        service.generate_speech("नमस्ते", "hi-IN", "shubh")


def test_empty_text_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        service.generate_speech("  ", "hi-IN", "aparna_hi_customer")


def test_text_over_2500_characters_is_rejected():
    with pytest.raises(ValueError, match="2,500 characters"):
        service.generate_speech("x" * 2501, "hi-IN", "aparna_hi_customer")


def test_speech_request_routes_to_correct_model_and_returns_wav(
    monkeypatch,
):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")

    expected_audio = b"RIFF-test-wav-data"
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "audios": [base64.b64encode(expected_audio).decode("ascii")]
            }

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(service.httpx, "post", fake_post)

    result = service.generate_speech(
        "வணக்கம்",
        language_code="ta-IN",
        speaker="ishita",
    )

    assert result == expected_audio
    assert captured["json"]["language_code"] == "ta-IN"
    assert captured["json"]["model"] == "bulbul:v3"
    assert captured["json"]["speaker"] == "ishita"
    assert captured["json"]["output_audio_codec"] == "wav"


def test_sarvam_empty_audio_response_is_reported(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"audios": []}

    monkeypatch.setattr(
        service.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(service.SarvamTTSUnavailableError, match="no audio"):
        service.generate_speech(
            "নমস্কার",
            language_code="bn-IN",
            speaker="roopa",
        )
