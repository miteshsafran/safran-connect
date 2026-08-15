from app.rag.embeddings import generate_embeddings


def main():

    texts = [
        "Employees are entitled to annual leave according to company policy.",
        "Employees can apply for leave through the HR portal.",
    ]

    print("Generating embeddings...")

    vectors = generate_embeddings(texts)

    print()
    print("Number of vectors:", len(vectors))

    for index, vector in enumerate(vectors):

        print()
        print(f"Vector {index + 1}")
        print("Dimensions:", len(vector))
        print("First 10 values:", vector[:10])


if __name__ == "__main__":
    main()