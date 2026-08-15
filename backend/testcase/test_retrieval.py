from app.rag.retriever import retrieve


def main():

    question = input(
        "Ask a question: "
    )

    print()
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(question)

    print()
    print("=" * 70)
    print("RETRIEVED DOCUMENTS")
    print("=" * 70)

    results = retrieve(
        question,
        limit=5,
    )

    if not results:
        print("No relevant documents found.")
        return

    for index, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(f"RESULT {index}")
        print("-" * 70)

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        metadata = result["metadata"]

        print(
            f"Source: "
            f"{metadata.get('source')}"
        )

        if metadata.get("page"):
            print(
                f"Page: "
                f"{metadata.get('page')}"
            )

        print(
            f"Chunk: "
            f"{metadata.get('chunk_id')}"
        )

        print()
        print("Text:")
        print(result["text"])


if __name__ == "__main__":
    main()