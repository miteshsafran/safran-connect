from typing import List, Dict

from langchain_ollama import ChatOllama

from app.config import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
)


llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    keep_alive=-1,
    num_predict=256,
)


SYSTEM_PROMPT = """
You are an internal company AI assistant.

Answer employee questions using ONLY the provided
company document context.

Rules:

1. Never invent information.
2. Do not use general knowledge when the answer
   is not present in the provided documents.
3. If the answer cannot be found in the documents,
   say that the information was not found.
4. Give a concise and clear answer.
5. Cite the source using [1], [2], [3], etc.
6. Only use citation numbers that exist in the context.
7. Do not create fake citations.
"""


def build_context(
    results: List[Dict],
) -> str:

    context_parts = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result["metadata"]

        source = metadata.get(
            "source",
            "Unknown",
        )

        page = metadata.get(
            "page",
            "Unknown",
        )

        chunk_id = metadata.get(
            "chunk_id",
            "Unknown",
        )

        text = result["text"]

        context_parts.append(
            f"""
SOURCE [{index}]
Document: {source}
Page: {page}
Chunk: {chunk_id}

Content:
{text}
"""
        )

    return "\n".join(context_parts)


def generate_answer(
    question: str,
    results: List[Dict],
) -> str:

    context = build_context(results)

    prompt = f"""
{SYSTEM_PROMPT}

EMPLOYEE QUESTION:
{question}

COMPANY DOCUMENT CONTEXT:

{context}

Now answer the employee question.

Include source citations such as [1] or [2]
after the relevant statements.

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content.strip()