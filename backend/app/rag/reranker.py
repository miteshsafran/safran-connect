from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):
        self.model = CrossEncoder(
            "BAAI/bge-reranker-v2-m3"
        )

    def rerank(
        self,
        question: str,
        results: list,
        top_k: int = 3,
    ):

        if not results:
            return []

        pairs = [
            [question, result["text"]]
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for result, score in zip(results, scores):

            item = result.copy()

            item["rerank_score"] = float(score)

            reranked.append(item)

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]