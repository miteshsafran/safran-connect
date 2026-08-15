from app.rag.loader import load_all_documents


documents = load_all_documents()

print()
print("=" * 60)
print(f"Total documents/pages loaded: {len(documents)}")
print("=" * 60)

for document in documents:
    print()
    print("SOURCE:", document["metadata"]["source"])
    print("TYPE:", document["metadata"]["file_type"])

    if "page" in document["metadata"]:
        print("PAGE:", document["metadata"]["page"])

    print("TEXT:")
    print(document["text"][:500])
    print("-" * 60)