
import pytest

from pydantic import ValidationError

from backend.app.services import qa_service

from backend.app.models.explanation import (
    CitedPoint,
    PolicyExplanation,
    PolicyExplanationDraft,
    PolicySection,
    SectionStatus,
)


def create_test_draft(source_number=1):
    available_section = PolicySection(
        status=SectionStatus.AVAILABLE,
        points=[
            CitedPoint(
                text="Eligible farmers receive Rs. 6000 annually.",
                source_numbers=[source_number],
            )
        ],
    )

    empty_section = PolicySection(
        status=SectionStatus.NOT_SPECIFIED,
        points=[],
    )

    return PolicyExplanationDraft(
        policy_overview=empty_section,
        eligibility=available_section,
        rights_and_benefits=available_section,
        obligations=empty_section,
        application_process=empty_section,
        deadlines_and_conditions=empty_section,
        real_world_impact=empty_section,
        information_not_found=[
            "Application deadlines were not specified."
        ],
    )


@pytest.fixture
def sample_chunk():
    return {
        "document_id": "sample-document-001",
        "filename": "sample_scheme.pdf",
        "page_number": 2,
        "language_label": "eng_Latn",
        "language_code": "eng",
        "script": "Latn",
        "detection_model": "IndicLID-FTR",
        "detection_raw_score": 0.95,
    }


def test_answer_question_builds_final_explanation(
    monkeypatch,
    sample_chunk,
):
    draft = create_test_draft()

    monkeypatch.setattr(
        qa_service,
        "search_documents",
        lambda question, top_k: [
            {
                **sample_chunk,
                "text": "Eligible farmers receive Rs. 6000 annually.",
            }
        ],
    )

    monkeypatch.setattr(
        qa_service,
        "generate_answer",
        lambda question, retrieved_chunks, language: draft,
    )

    result = qa_service.answer_question(
        question="What assistance do farmers receive?",
        language="Hindi",
    )

    assert isinstance(result, PolicyExplanation)

    assert result.answer_language == "Hindi"

    assert len(result.sources) == 1

    assert result.sources[0].filename == "sample_scheme.pdf"

    assert result.sources[0].language_label == "eng_Latn"

    assert result.eligibility.points[0].source_numbers == [1]


def test_answer_question_rejects_invalid_source_numbers(
    monkeypatch,
    sample_chunk,
):
    draft = create_test_draft(source_number=2)

    monkeypatch.setattr(
        qa_service,
        "search_documents",
        lambda question, top_k: [
            {
                **sample_chunk,
                "text": "Eligible farmers receive Rs. 6000 annually.",
            }
        ],
    )

    monkeypatch.setattr(
        qa_service,
        "generate_answer",
        lambda question, retrieved_chunks, language: draft,
    )

    with pytest.raises(ValidationError):
        qa_service.answer_question(
            question="What assistance do farmers receive?",
            language="English",
        )


def test_answer_question_handles_no_retrieved_sources(
    monkeypatch,
):
    monkeypatch.setattr(
        qa_service,
        "search_documents",
        lambda question, top_k: [],
    )

    from backend.app.services.gemini_service import (
        create_empty_explanation,
    )

    monkeypatch.setattr(
        qa_service,
        "generate_answer",
        lambda question, retrieved_chunks, language: (
            create_empty_explanation()
        ),
    )

    result = qa_service.answer_question(
        question="What is this policy about?",
        language="English",
    )

    assert isinstance(result, PolicyExplanation)
    assert result.sources == []
    assert result.policy_overview.status == SectionStatus.NOT_SPECIFIED
