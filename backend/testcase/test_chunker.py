from app.rag.loader import load_all_documents
from app.rag.chunker import chunk_documents


documents = load_all_documents()

print(f"Loaded pages: {len(documents)}")

chunks = chunk_documents(documents)

print(f"Generated chunks: {len(chunks)}")

print()
print("=" * 60)

for index, chunk in enumerate(chunks[:10], start=1):

    print(f"CHUNK {index}")

    print("Source:", chunk["metadata"]["source"])

    if "page" in chunk["metadata"]:
        print("Page:", chunk["metadata"]["page"])

    print("Chunk ID:", chunk["metadata"]["chunk_id"])

    print()
    print(chunk["text"])

    print("=" * 60)