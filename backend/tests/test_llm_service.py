import pytest

from backend.app.services import llm_service
from backend.app.services.sarvam_llm_service import SarvamLLMUnavailableError


def test_sarvam_is_default_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    sentinel = object()

    monkeypatch.setattr(
        llm_service,
        "generate_sarvam_answer",
        lambda **kwargs: sentinel,
    )

    assert llm_service.generate_answer("Q", [{"text": "evidence"}]) is sentinel


def test_gemini_can_be_selected_directly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    sentinel = object()

    monkeypatch.setattr(
        llm_service,
        "generate_gemini_answer",
        lambda **kwargs: sentinel,
    )

    assert llm_service.generate_answer("Q", [{"text": "evidence"}]) is sentinel


def test_sarvam_failure_falls_back_to_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "sarvam")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")

    monkeypatch.setattr(
        llm_service,
        "generate_sarvam_answer",
        lambda **kwargs: (_ for _ in ()).throw(SarvamLLMUnavailableError("test outage")),
    )

    sentinel = object()
    monkeypatch.setattr(
        llm_service,
        "generate_gemini_answer",
        lambda **kwargs: sentinel,
    )

    assert llm_service.generate_answer("Q", [{"text": "evidence"}]) is sentinel


def test_sarvam_failure_does_not_fallback_when_disabled(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "sarvam")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "false")

    monkeypatch.setattr(
        llm_service,
        "generate_sarvam_answer",
        lambda **kwargs: (_ for _ in ()).throw(SarvamLLMUnavailableError("test outage")),
    )

    with pytest.raises(SarvamLLMUnavailableError, match="test outage"):
        llm_service.generate_answer("Q", [{"text": "evidence"}])


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        llm_service.generate_answer("Q", [{"text": "evidence"}])
