import json

import httpx
import pytest

from backend.app.models.explanation import PolicyExplanationDraft
from backend.app.services import sarvam_llm_service as service


def valid_draft_payload():
    empty_section = {"status": "not_specified", "points": []}
    return {
        "policy_overview": empty_section,
        "eligibility": {
            "status": "available",
            "points": [
                {
                    "text": "Eligible small businesses may apply.",
                    "source_numbers": [1],
                    "evidence_basis": "directly_stated",
                }
            ],
        },
        "rights_and_benefits": empty_section,
        "obligations": empty_section,
        "application_process": empty_section,
        "deadlines_and_conditions": empty_section,
        "real_world_impact": empty_section,
        "information_not_found": [],
    }


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.is_error = status_code >= 400
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def test_sarvam_request_uses_schema_and_validates_response(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    captured = {}
    draft_json = valid_draft_payload()

    def fake_post(url, *, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "payload": json, "timeout": timeout})
        return FakeResponse({
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": __import__("json").dumps(draft_json)},
            }]
        })

    monkeypatch.setattr(service.httpx, "post", fake_post)

    result = service.generate_sarvam_answer(
        question="Who can apply?",
        retrieved_chunks=[{
            "filename": "policy.pdf",
            "page_number": 2,
            "language_label": "eng_Latn",
            "text": "Eligible small businesses may apply.",
        }],
        language="Hindi",
    )

    assert isinstance(result, PolicyExplanationDraft)
    assert result.eligibility.points[0].source_numbers == [1]
    assert captured["headers"]["api-subscription-key"] == "test-key"
    assert captured["payload"]["model"] == "sarvam-105b"
    assert captured["payload"]["response_format"]["type"] == "json_schema"
    schema = captured["payload"]["response_format"]["json_schema"]["schema"]
    assert "information_not_found" in schema["required"]
    assert "evidence_basis" in schema["$defs"]["CitedPoint"]["required"]
    assert "points" in schema["$defs"]["PolicySection"]["required"]
    assert captured["payload"]["messages"][1]["content"].find("PREFERRED ANSWER LANGUAGE") >= 0


def test_missing_sarvam_key_is_actionable(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)

    with pytest.raises(service.SarvamLLMUnavailableError, match="SARVAM_API_KEY"):
        service.generate_sarvam_answer(
            "Who can apply?",
            [{"text": "Some evidence"}],
            "English",
        )


def test_empty_retrieval_does_not_call_sarvam(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    monkeypatch.setattr(
        service.httpx,
        "post",
        lambda *args, **kwargs: pytest.fail("API must not be called without evidence"),
    )

    result = service.generate_sarvam_answer("Question", [], "English")
    assert isinstance(result, PolicyExplanationDraft)
    assert result.sources == [] if hasattr(result, "sources") else True


def test_http_overload_is_reported_and_retried_once(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        return FakeResponse({"error": {"message": "model overloaded"}}, 503)

    monkeypatch.setattr(service.httpx, "post", fake_post)
    monkeypatch.setattr(service.time, "sleep", lambda _: None)

    with pytest.raises(service.SarvamLLMUnavailableError, match="HTTP 503"):
        service.generate_sarvam_answer(
            "Who can apply?",
            [{"text": "Eligible businesses may apply."}],
            "English",
        )

    assert calls["count"] == 2
