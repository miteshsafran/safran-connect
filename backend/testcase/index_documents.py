from app.rag.loader import load_all_documents
from app.rag.chunker import chunk_documents
from app.rag.embeddings import generate_embeddings
from app.rag.vector_store import create_collection, insert_chunks


def main():

    print("=" * 60)
    print("COMPANY DOCUMENT INDEXING")
    print("=" * 60)

    # --------------------------------
    # 1. Load documents
    # --------------------------------

    print("\n1. Loading documents...")

    documents = load_all_documents()

    print(
        f"Loaded pages/documents: "
        f"{len(documents)}"
    )

    if not documents:
        print("No documents found.")
        return

    # --------------------------------
    # 2. Chunk documents
    # --------------------------------

    print("\n2. Creating chunks...")

    chunks = chunk_documents(documents)

    print(
        f"Created chunks: "
        f"{len(chunks)}"
    )

    if not chunks:
        print("No chunks created.")
        return

    # --------------------------------
    # 3. Generate embeddings
    # --------------------------------

    print("\n3. Generating embeddings...")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    vectors = generate_embeddings(texts)

    print(
        f"Generated vectors: "
        f"{len(vectors)}"
    )

    if not vectors:
        print("No embeddings generated.")
        return

    # --------------------------------
    # 4. Create Qdrant collection
    # --------------------------------

    print("\n4. Creating Qdrant collection...")

    vector_size = len(vectors[0])

    print(
        f"Embedding vector dimension: "
        f"{vector_size}"
    )

    create_collection(vector_size)

    # --------------------------------
    # 5. Store vectors
    # --------------------------------

    print("\n5. Storing vectors in Qdrant...")

    insert_chunks(
        chunks,
        vectors,
    )

    print()
    print("=" * 60)
    print("INDEXING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()