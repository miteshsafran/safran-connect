import time

from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse
from app.rag.retriever import retrieve
from app.rag.generator import generate_answer
from app.rag.citations import build_sources


router = APIRouter(
    prefix="/api",
    tags=["Chat"],
)


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    total_start = time.perf_counter()

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # -----------------------------
    # Retrieval
    # -----------------------------

    start = time.perf_counter()

    results = retrieve(
        question=question,
        limit=10,
    )

    retrieval_time = time.perf_counter() - start

    if not results:
        return ChatResponse(
            answer=(
                "I could not find relevant information "
                "in the company documents."
            ),
            sources=[],
        )

    # -----------------------------
    # LLM generation
    # -----------------------------

    start = time.perf_counter()

    answer = generate_answer(
        question=question,
        results=results,
    )

    generation_time = time.perf_counter() - start

    # -----------------------------
    # Sources
    # -----------------------------

    sources = build_sources(results)

    total_time = time.perf_counter() - total_start

    print()
    print("=" * 50)
    print("CHAT PERFORMANCE")
    print("=" * 50)
    print(f"Retrieval:   {retrieval_time:.2f}s")
    print(f"Generation:  {generation_time:.2f}s")
    print(f"Total:       {total_time:.2f}s")
    print("=" * 50)

    return ChatResponse(
        answer=answer,
        sources=sources,
    )