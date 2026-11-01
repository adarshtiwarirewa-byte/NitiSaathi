
from datetime import date

import pytest
from pydantic import ValidationError

from backend.app.models.policy import (
    PolicyDocument,
    JurisdictionLevel,
    DocumentType,
    DocumentStatus,
    VerificationStatus,
)


def test_valid_policy_document():

    document = PolicyDocument(
        original_filename="sample_scheme.pdf",
        title="Small Business Scheme",
        document_type=DocumentType.SCHEME,
        jurisdiction_level=JurisdictionLevel.STATE,
        state="Madhya Pradesh",
        effective_date=date(2026, 1, 1),
        effective_until=date(2026, 12, 31),
    )

    assert document.title == "Small Business Scheme"
    assert document.state == "Madhya Pradesh"
    assert document.jurisdiction_level == JurisdictionLevel.STATE


def test_invalid_jurisdiction_rejected():

    with pytest.raises(ValidationError):

        PolicyDocument(
            original_filename="sample.pdf",
            jurisdiction_level="national"
        )


def test_invalid_effective_dates_rejected():

    with pytest.raises(ValidationError):

        PolicyDocument(
            original_filename="sample.pdf",
            effective_date=date(2026, 12, 31),
            effective_until=date(2026, 1, 1),
        )


def test_document_id_generated_automatically():

    document1 = PolicyDocument(
        original_filename="scheme1.pdf"
    )

    document2 = PolicyDocument(
        original_filename="scheme2.pdf"
    )

    assert document1.document_id
    assert document2.document_id
    assert document1.document_id != document2.document_id


def test_default_policy_statuses():

    document = PolicyDocument(
        original_filename="new_policy.pdf"
    )

    assert document.document_type == DocumentType.UNKNOWN
    assert document.jurisdiction_level == JurisdictionLevel.UNKNOWN
    assert document.document_status == DocumentStatus.UNKNOWN
    assert document.verification_status == VerificationStatus.UNVERIFIED


def test_source_languages_default_to_empty_list():

    document = PolicyDocument(
        original_filename="sample.pdf"
    )

    assert document.source_languages == []
