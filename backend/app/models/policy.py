
from datetime import date
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator
from enum import Enum


class JurisdictionLevel(str, Enum):
    UNKNOWN = "unknown"
    CENTRAL = "central"
    STATE = "state"
    MUNICIPAL = "municipal"


class DocumentType(str, Enum):
    UNKNOWN = "unknown"
    SCHEME = "scheme"
    NOTIFICATION = "notification"
    POLICY = "policy"
    MUNICIPAL_LAW = "municipal_law"
    GUIDELINES = "guidelines"
    AMENDMENT = "amendment"
    CORRIGENDUM = "corrigendum"
    OTHER = "other"


class DocumentStatus(str, Enum):
    UNKNOWN = "unknown"
    DRAFT = "draft"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REPEALED = "repealed"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    OFFICIAL_SOURCE_VERIFIED = "official_source_verified"



class PolicyDocument(BaseModel):
    """
    Represents one government policy source document
    and its official metadata.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True
    )

    # Document identity
    document_id: str = Field(
        default_factory=lambda: str(uuid4())
    )
    original_filename: str = Field(
        min_length=1,
        max_length=255
    )
    title: str | None = Field(
        default=None,
        max_length=500
    )

    # Document classification
    document_type: DocumentType = DocumentType.UNKNOWN
    issuing_authority: str | None = Field(
        default=None,
        max_length=300
    )

    # Geographic applicability
    jurisdiction_level: JurisdictionLevel = (
        JurisdictionLevel.UNKNOWN
    )
    state: str | None = Field(default=None, max_length=100)
    district_or_city: str | None = Field(
        default=None,
        max_length=150
    )

    # Publication and validity
    publication_date: date | None = None
    effective_date: date | None = None
    effective_until: date | None = None

    # Language and source information
    source_languages: list[str] = Field(
        default_factory=list,
        description="Languages detected in the original document"
    )
    source_url: HttpUrl | None = None

    # Version and legal status
    version: str | None = Field(
        default=None,
        max_length=50
    )
    document_status: DocumentStatus = DocumentStatus.UNKNOWN
    verification_status: VerificationStatus = (
        VerificationStatus.UNVERIFIED
    )
    supersedes_document_id: str | None = None

    # Extraction information
    page_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_effective_dates(self):
        if (
            self.effective_date is not None
            and self.effective_until is not None
            and self.effective_until < self.effective_date
        ):
            raise ValueError(
                "Effective end date cannot be earlier than "
                "effective start date."
            )
        return self
