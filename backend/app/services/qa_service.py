from backend.app.services.retrieval import search_documents
from backend.app.services.gemini_service import generate_answer


def answer_question(
    question: str,
    language: str = "English",
    top_k: int = 3
) -> dict:

    # Step 1: Retrieve relevant passages
    retrieved_chunks = search_documents(
        question=question,
        top_k=top_k
    )

    # Step 2: Generate an answer using retrieved evidence
    answer = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        language=language
    )

    # Step 3: Prepare source references
    sources = []

    for chunk in retrieved_chunks:
        sources.append({
            "filename": chunk["filename"],
            "page_number": chunk["page_number"]
        })

    return {
        "question": question,
        "answer": answer,
        "language": language,
        "sources": sources
    }