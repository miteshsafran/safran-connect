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

Answer the employee's question using ONLY the provided CONTEXT.

RULES:

1. Answer ONLY what the employee asked.
2. Do not add unrelated information from the context.
3. Do not use general knowledge or assumptions.
4. Never invent information.
5. If the answer is not clearly available in the context, say:
   "I couldn't find this information in the available company documents."
6. For simple questions, give a short answer in 1-2 sentences.
7. If the question asks for a number, date, limit, or yes/no answer,
   give that answer directly.
8. Do not summarize the entire policy section.
9. Only mention additional rules if they are directly required
   to answer the question.
10. Every factual statement must have a valid citation.
11. Use ONLY citation numbers that exist in the CONTEXT.
12. Never create fake citations.

CITATION FORMAT:

Example:
"Employees are entitled to 12 casual leave days per calendar year. [1]"

At the end:

Sources:
[1] Leave Policy.pdf - Page 5

FINAL CHECK:

Before answering, identify exactly what the employee asked.
Answer only that question.

Accuracy is more important than completeness.
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