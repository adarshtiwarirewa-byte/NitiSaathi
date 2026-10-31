from fastapi import APIRouter, UploadFile, File

from backend.app.services.ingestion import ingest_pdf
from pydantic import BaseModel, Field
from backend.app.services.retrieval import search_documents
from backend.app.services.qa_service import answer_question


router = APIRouter()

class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    language: str = Field(default="English", max_length=50)
    top_k: int = Field(default=3, ge=1, le=10)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a government policy PDF
    and extract its page-wise text.
    """

    result = await ingest_pdf(file)

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
    result = answer_question(
        question=request.question,
        language=request.language,
        top_k=request.top_k
    )

    return {
        "success": True,
        "result": result
    }