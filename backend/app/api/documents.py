from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from backend.app.services.ingestion import ingest_pdf
from pydantic import BaseModel, Field
from backend.app.services.retrieval import search_documents
from backend.app.services.qa_service import answer_question

from backend.app.services.gemini_service import (
    GeminiServiceUnavailableError,
)

router = APIRouter()

class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    language: str = Field(default="English", max_length=50)
    top_k: int = Field(default=3, ge=1, le=10)


@router.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Upload a government policy PDF,
    extract its text and identify its languages.
    """

    indiclid_service = request.app.state.indiclid_service

    result = await ingest_pdf(
        file,
        indiclid_service=indiclid_service
    )

    return {
        "success": True,
        "message": "Document uploaded and processed successfully.",
        "document": result,
    }

@router.post("/search")
def search_document(request: SearchRequest):
    results = search_documents(
        question=request.question,
        top_k=request.top_k
    )

    return {
        "success": True,
        "question": request.question,
        "results_count": len(results),
        "results": results
    }


@router.post("/ask")
def ask_question(request: AskRequest):

    try:
        result = answer_question(
            question=request.question,
            language=request.language,
            top_k=request.top_k
        )

    except GeminiServiceUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc

    return {
        "success": True,
        "result": result
    }