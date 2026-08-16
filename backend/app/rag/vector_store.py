from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from app.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
)


client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
)


def collection_exists() -> bool:
    """
    Check whether our Qdrant collection exists.
    """

    collections = client.get_collections()

    return any(
        collection.name == QDRANT_COLLECTION
        for collection in collections.collections
    )


def create_collection(vector_size: int) -> None:
    """
    Create the company documents collection.
    """

    if collection_exists():
        print(
            f"Collection already exists: "
            f"{QDRANT_COLLECTION}"
        )
        return

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Created collection: "
        f"{QDRANT_COLLECTION}"
    )


def insert_chunks(
    chunks: List[Dict],
    vectors: List[List[float]],
) -> None:
    """
    Insert document chunks and embeddings into Qdrant.
    """

    if not chunks:
        return

    if len(chunks) != len(vectors):
        raise ValueError(
            "Number of chunks and vectors must match."
        )

    points = []

    for index, (chunk, vector) in enumerate(
        zip(chunks, vectors)
    ):

        payload = {
            "text": chunk["text"],
            **chunk["metadata"],
        }

        points.append(
            PointStruct(
                id=index,
                vector=vector,
                payload=payload,
            )
        )

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
    )

    print(
        f"Inserted {len(points)} chunks into Qdrant."
    )

def search(
    query_vector: List[float],
    limit: int = 10,
) -> List[Dict]:
    """
    Search Qdrant for the most relevant document chunks.
    """

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    ).points

    matches = []

    for result in results:

        payload = result.payload or {}

        matches.append(
            {
                "score": result.score,
                "text": payload.get("text", ""),
                "metadata": {
                    key: value
                    for key, value in payload.items()
                    if key != "text"
                },
            }
        )

    return matches