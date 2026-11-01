
from backend.app.services.gemini_service import (
    generate_answer,
    create_empty_explanation,
)

from backend.app.models.explanation import (
    PolicyExplanationDraft,
    SectionStatus,
)
import pytest

def test_empty_retrieval_returns_structured_explanation():
    """No retrieved evidence should not trigger Gemini."""

    explanation = create_empty_explanation()

    assert isinstance(explanation, PolicyExplanationDraft)

    assert (
        explanation.eligibility.status
        == SectionStatus.NOT_SPECIFIED
    )

    assert (
        explanation.rights_and_benefits.status
        == SectionStatus.NOT_SPECIFIED
    )

    assert explanation.information_not_found


@pytest.mark.integration
def test_gemini_returns_structured_policy_explanation():
    """
    Live Gemini integration test.

    Requires GEMINI_API_KEY in the project .env file.
    """

    question = "What financial assistance do farmers receive?"

    retrieved_chunks = [
        {
            "document_id": "sample-document-001",
            "filename": "sample_scheme.pdf",
            "page_number": 2,
            "language_label": "eng_Latn",
            "text": (
                "Eligible farmers receive financial assistance "
                "of Rs. 6000 per year in three installments."
            )
        }
    ]

    explanation = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        language="English"
    )

    assert isinstance(explanation, PolicyExplanationDraft)

    sections = [
        explanation.policy_overview,
        explanation.eligibility,
        explanation.rights_and_benefits,
        explanation.obligations,
        explanation.application_process,
        explanation.deadlines_and_conditions,
        explanation.real_world_impact,
    ]

    # At least one section must contain supported information.
    assert any(
        section.status == SectionStatus.AVAILABLE
        for section in sections
    )

    # Every generated factual point must cite source 1.
    for section in sections:
        for point in section.points:
            assert point.source_numbers
            assert all(
                number == 1
                for number in point.source_numbers
            )

    print("\nStructured Gemini explanation:")
    print(explanation.model_dump_json(indent=2))

    print("\nGemini structured-output test passed!")
