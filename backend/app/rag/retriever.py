import time

from typing import List, Dict

from app.rag.embeddings import generate_query_embedding
from app.rag.vector_store import search


def retrieve(
    question: str,
    limit: int = 3,
) -> List[Dict]:

    start = time.perf_counter()

    query_vector = generate_query_embedding(
        question
    )

    embedding_time = time.perf_counter() - start

    start = time.perf_counter()

    results = search(
        query_vector=query_vector,
        limit=limit,
    )

    qdrant_time = time.perf_counter() - start

    print(
        f"Embedding time: {embedding_time:.2f}s"
    )

    print(
        f"Qdrant time: {qdrant_time:.2f}s"
    )

    return results