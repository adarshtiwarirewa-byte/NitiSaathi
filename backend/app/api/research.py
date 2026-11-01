"""Standalone web-research API. Does not alter document endpoints."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.services.web_research_service import (
    WebResearchUnavailableError,
    research_policy,
)

router = APIRouter()


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    language: str = Field(default="English", min_length=2, max_length=50)
    top_k: int = Field(default=6, ge=1, le=8)


@router.post("/ask")
def ask_web_research(request: ResearchRequest):
    """Search reliable web sources when the user has not selected a PDF."""
    try:
        result = research_policy(
            question=request.question,
            language=request.language,
            top_k=request.top_k,
        )
    except (WebResearchUnavailableError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"success": True, "result": result}
