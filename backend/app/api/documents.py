import json
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, Response

from backend.app.services.ingestion import ingest_pdf
from pydantic import BaseModel, Field
from backend.app.services.retrieval import search_documents
from backend.app.services.qa_service import answer_question
from backend.app.services.indiclid_service import IndicLIDService

from backend.app.services.gemini_service import (
    GeminiServiceUnavailableError,
)
from backend.app.services.sarvam_tts_service import (
    SarvamTTSUnavailableError,
    generate_speech,
)

router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    language: str = Field(default="English", max_length=50)
    top_k: int = Field(default=3, ge=1, le=10)
    document_id: str | None = Field(default=None, max_length=100)


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2500)
    language_code: str = Field(..., pattern="^(en-IN|hi-IN|bn-IN|gu-IN|kn-IN|mr-IN|pa-IN|ta-IN|te-IN)$")
    speaker: str = Field(..., min_length=3, max_length=80)


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
    if indiclid_service is None:
        try:
            indiclid_service = IndicLIDService()
            request.app.state.indiclid_service = indiclid_service
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="The document language detector could not start. Check that its model files are installed.",
            ) from exc

    result = await ingest_pdf(
        file,
        indiclid_service=indiclid_service
    )

    return {
        "success": True,
        "message": "Document uploaded and processed successfully.",
        "document": result,
    }


@router.get("")
def list_documents():
    """List documents already uploaded in this local demo."""
    documents = []
    if not PROCESSED_DIR.exists():
        return {"documents": documents}
    for path in PROCESSED_DIR.glob("*.json"):
        if path.name.endswith("_chunks.json"):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            documents.append({
                "document_id": record["document_id"],
                "filename": record.get("original_filename", "Policy document"),
                "page_count": record.get("page_count", len(record.get("pages", []))),
            })
        except (OSError, ValueError, KeyError):
            continue
    return {"documents": sorted(documents, key=lambda item: item["filename"].lower())}


@router.get("/{document_id}/pages/{page_number}")
def get_document_page(document_id: str, page_number: int):
    """Return extracted page text for a citation preview."""
    path = PROCESSED_DIR / f"{document_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found.")
    record = json.loads(path.read_text(encoding="utf-8"))
    page = next((p for p in record.get("pages", []) if p["page_number"] == page_number), None)
    if page is None:
        raise HTTPException(status_code=404, detail="Page not found.")
    return {"filename": record.get("original_filename"), **page}


@router.get("/{document_id}/file")
def get_document_file(document_id: str):
    path = UPLOAD_DIR / f"{document_id}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document file not found.")
    record_path = PROCESSED_DIR / f"{document_id}.json"
    filename = "policy.pdf"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        filename = record.get("original_filename", filename)
    return FileResponse(path, media_type="application/pdf", filename=filename)

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
            top_k=request.top_k,
            document_id=request.document_id,
        )

    except (GeminiServiceUnavailableError, ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc) or "The explanation service is unavailable."
        ) from exc

    return {
        "success": True,
        "result": result
    }


@router.post("/speak")
def speak_explanation(request: SpeakRequest):
    """Return a WAV narration of an existing plain-language answer."""
    try:
        audio = generate_speech(
            request.text,
            language_code=request.language_code,
            speaker=request.speaker,
        )
    except (SarvamTTSUnavailableError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=niti-saathi-explanation.wav"},
    )
