
import pytest
from pydantic import ValidationError

from backend.app.models.explanation import (
    PolicyExplanation,
    PolicySection,
    SectionStatus,
)


def test_available_section_requires_points():
    with pytest.raises(ValidationError):
        PolicySection(
            status=SectionStatus.AVAILABLE,
            points=[]
        )


def test_not_specified_section_cannot_have_points():
    with pytest.raises(ValidationError):
        PolicySection(
            status=SectionStatus.NOT_SPECIFIED,
            points=[
                {
                    "text": "Some policy information.",
                    "source_numbers": [1]
                }
            ]
        )


def test_explanation_rejects_unknown_source_numbers():
    section = {
        "status": "available",
        "points": [
            {
                "text": "Eligible businesses may apply.",
                "source_numbers": [2]
            }
        ]
    }

    empty_section = {
        "status": "not_specified",
        "points": []
    }

    with pytest.raises(ValidationError):
        PolicyExplanation(
            question="Who can apply?",
            answer_language="English",
            policy_overview=section,
            eligibility=section,
            rights_and_benefits=empty_section,
            obligations=empty_section,
            application_process=empty_section,
            deadlines_and_conditions=empty_section,
            real_world_impact=empty_section,
            sources=[
                {
                    "source_number": 1,
                    "document_id": "doc-123",
                    "filename": "policy.pdf",
                    "page_number": 1
                }
            ]
        )
