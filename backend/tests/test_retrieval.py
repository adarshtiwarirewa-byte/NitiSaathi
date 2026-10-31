
from backend.app.services.retrieval import search_documents


def test_retrieval():

    questions = [
        "What financial assistance do farmers receive?",
        "किसानों को सरकार से क्या आर्थिक सहायता मिलती है?"
    ]

    for question in questions:

        print("\n" + "=" * 60)
        print("QUESTION:", question)

        results = search_documents(
            question=question,
            top_k=2
        )

        print("\nRetrieved chunks:", len(results))

        for index, result in enumerate(results, start=1):

            print(f"\nResult {index}")
            print("Filename:", result["filename"])
            print("Page:", result["page_number"])
            print("Distance:", result["distance"])
            print("Text:", result["text"][:300])

        assert len(results) > 0

    print("\nAll retrieval tests passed!")


if __name__ == "__main__":
    test_retrieval()
