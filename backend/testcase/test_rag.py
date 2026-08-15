from app.rag.retriever import retrieve
from app.rag.generator import generate_answer


def main():

    question = input(
        "Ask a company question: "
    )

    print()
    print("=" * 70)
    print("SEARCHING COMPANY DOCUMENTS...")
    print("=" * 70)

    results = retrieve(
        question,
        limit=5,
    )

    if not results:

        print(
            "I could not find relevant information "
            "in the company documents."
        )

        return

    print(
        f"Retrieved {len(results)} relevant chunks."
    )

    print()
    print("=" * 70)
    print("GENERATING ANSWER...")
    print("=" * 70)

    answer = generate_answer(
        question=question,
        results=results,
    )

    print()
    print("ANSWER:")
    print(answer)

    print()
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result["metadata"]

        print(
            f"{index}. "
            f"{metadata.get('source')}"
            f" - Page "
            f"{metadata.get('page', 'N/A')}"
            f" - Score "
            f"{result['score']:.4f}"
        )


if __name__ == "__main__":
    main()