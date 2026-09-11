
from sentence_transformers import CrossEncoder


DEFAULT_RERANKER = (
    "BAAI/bge-reranker-base"
)


class Reranker:

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
    ):

        self.model_name = model_name

        self.model = CrossEncoder(
            model_name,
            max_length=512,
        )

    def rerank(
        self,
        query: str,
        candidates,
        top_k: int = 5,
    ):

        if not candidates:
            return []

        pairs = [
            (
                query,
                result["chunk"]["content"],
            )
            for result in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        reranked = []

        for result, score in zip(
            candidates,
            scores,
        ):

            reranked.append({
                "chunk_id": result["chunk_id"],
                "rank": None,

                # Keep RRF score for traceability
                "rrf_score": result[
                    "rrf_score"
                ],

                "reranker_score": float(
                    score
                ),

                "chunk": result["chunk"],
            })

        reranked.sort(
            key=lambda x: x[
                "reranker_score"
            ],
            reverse=True,
        )

        for rank, result in enumerate(
            reranked,
            start=1,
        ):

            result["rank"] = rank

        return reranked[:top_k]
