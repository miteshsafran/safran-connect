from typing import List, Dict


def build_sources(
    results: List[Dict],
) -> List[Dict]:

    sources = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result["metadata"]

        sources.append(
            {
                "id": index,
                "document": metadata.get(
                    "source",
                    "Unknown",
                ),
                "page": metadata.get("page"),
                "chunk_id": metadata.get(
                    "chunk_id"
                ),
                "score": round(
                    float(result["score"]),
                    4,
                ),
                "text": result["text"],
            }
        )

    return sources