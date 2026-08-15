from typing import List

from langchain_ollama import OllamaEmbeddings

from app.config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
)


embedding_model = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
    keep_alive=-1,
)


def generate_embeddings(
    texts: List[str],
) -> List[List[float]]:
    """
    Generate embeddings for multiple document chunks.
    """

    if not texts:
        return []

    return embedding_model.embed_documents(texts)


def generate_query_embedding(
    text: str,
) -> List[float]:
    """
    Generate embedding for a user question.
    """

    return embedding_model.embed_query(text)