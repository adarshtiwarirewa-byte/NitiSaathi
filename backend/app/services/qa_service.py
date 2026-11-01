from backend.app.services.retrieval import search_documents
from backend.app.services.llm_service import generate_answer

from backend.app.models.explanation import (
    EvidenceReference,
    PolicyExplanation,
)


def answer_question(
    question: str,
    language: str = "English",
    top_k: int = 3,
    document_id: str | None = None,
) -> PolicyExplanation:
    """Retrieve policy evidence, generate an explanation, and attach citations."""

    # Step 1: Retrieve relevant passages.
    search_args = {"question": question, "top_k": top_k}
    if document_id:
        search_args["document_id"] = document_id
    retrieved_chunks = search_documents(**search_args)

    # Step 2: Generate a structured explanation using the configured LLM.
    draft = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        language=language,
    )

    # Step 3: Construct source references using actual backend data.
    sources = []
    for source_number, chunk in enumerate(retrieved_chunks, start=1):
        sources.append(
            EvidenceReference(
                source_number=source_number,
                document_id=chunk["document_id"],
                filename=chunk["filename"],
                page_number=chunk["page_number"],
                language_label=chunk.get("language_label"),
                language_code=chunk.get("language_code"),
                script=chunk.get("script"),
                detection_model=chunk.get("detection_model"),
                detection_raw_score=chunk.get("detection_raw_score"),
            )
        )

    # Step 4: Validate the final explanation and every citation.
    explanation = PolicyExplanation(
        question=question,
        answer_language=language,
        **draft.model_dump(),
        sources=sources,
    )

    return explanation
