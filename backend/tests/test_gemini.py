from backend.app.services.gemini_service import generate_answer


def test_gemini():
    question = "What financial assistance do farmers receive?"

    retrieved_chunks = [
        {
            "filename": "sample_scheme.pdf",
            "page_number": 2,
            "text": (
                "Eligible farmers receive financial assistance "
                "of Rs. 6000 per year in three installments."
            )
        }
    ]

    answer = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        language="English"
    )

    print("\nGemini Generated Answer:")
    print(answer)

    assert answer
    print("\nGemini test passed!")


if __name__ == "__main__":
    test_gemini()