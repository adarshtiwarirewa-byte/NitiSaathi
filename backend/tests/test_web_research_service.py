from types import SimpleNamespace

from backend.app.services.web_research_service import (
    _classify_url,
    _grounded_sources,
)


def test_classifies_official_government_domains():
    assert _classify_url("https://pmkisan.gov.in/") == (
        "pmkisan.gov.in", "official_government"
    )
    assert _classify_url("https://www.india.gov.in/") == (
        "www.india.gov.in", "official_government"
    )


def test_classifies_reputable_publication_as_supporting():
    assert _classify_url("https://www.thehindu.com/news/") == (
        "www.thehindu.com", "reputable_publication"
    )


def test_rejects_unknown_and_insecure_domains():
    assert _classify_url("http://example.gov.in") is None
    assert _classify_url("https://unknown-example.com") is None


def test_extracts_only_sources_with_grounded_segments():
    response = SimpleNamespace(candidates=[SimpleNamespace(
        grounding_metadata=SimpleNamespace(
            grounding_chunks=[
                SimpleNamespace(web=SimpleNamespace(
                    uri="https://pmkisan.gov.in/", title="PM-KISAN"
                )),
                SimpleNamespace(web=SimpleNamespace(
                    uri="https://unknown-example.com/", title="Unknown"
                )),
            ],
            grounding_supports=[
                SimpleNamespace(
                    segment=SimpleNamespace(text="Eligibility evidence."),
                    grounding_chunk_indices=[0],
                )
            ],
        )
    )])

    chunks, sources = _grounded_sources(response)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Eligibility evidence."
    assert chunks[0]["source_url"] == "https://pmkisan.gov.in/"
    assert sources[0]["source_number"] == 1
