import time

from typing import List, Dict

from app.rag.embeddings import generate_query_embedding
from app.rag.vector_store import search
from app.rag.reranker import Reranker


# Tune this later based on your embedding model
RELEVANCE_THRESHOLD = 0.45

reranker = Reranker()

def retrieve(
    question: str,
    limit: int = 10,
) -> List[Dict]:

    start = time.perf_counter()
    # question embedding
    query_vector = generate_query_embedding(
        question
    )

    embedding_time = time.perf_counter() - start

    start = time.perf_counter()
    # Search Context from vector db
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

    # ---------------------------------
    # Remove irrelevant results
    # ---------------------------------

    relevant_results = [
        result
        for result in results
        if result["score"] >= RELEVANCE_THRESHOLD
    ]

    print(
        f"Relevant candidates: "
        f"{len(relevant_results)} / {len(results)}"
    )

    if not relevant_results:
        return []

    # -----------------------------
    # Reranking
    # -----------------------------

    start = time.perf_counter()

    reranked_results = reranker.rerank(
        question=question,
        results=relevant_results,
        top_k=3,
    )

    reranker_time = time.perf_counter() - start

    print(
        f"Reranker time: {reranker_time:.2f}s"
    )

    # -----------------------------
    # Debug scores
    # -----------------------------

    for result in reranked_results:

        print(
            f"Qdrant: {result['score']:.4f} | "
            f"Rerank: {result['rerank_score']:.4f} | "
            f"Document: {result.get('document')}"
        )

    return reranked_results