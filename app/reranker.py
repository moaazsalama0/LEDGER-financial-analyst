
from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker(
    model_name=DEFAULT_RERANKER_MODEL
):
    return CrossEncoder(model_name)


def rerank_results(
    query,
    candidates,
    reranker,
    top_k=5
):
    if not query or not str(query).strip():
        raise ValueError("Query must not be empty.")

    if reranker is None:
        raise ValueError("Reranker must not be None.")

    if not candidates:
        return []

    pairs = [
        [str(query), str(candidate.get("content", ""))]
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    reranked = []

    for candidate, score in zip(candidates, scores):
        result = candidate.copy()
        result["reranker_score"] = float(score)
        reranked.append(result)

    reranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    final_k = min(
        max(int(top_k), 1),
        len(reranked)
    )

    final_results = []

    for rank, result in enumerate(
        reranked[:final_k],
        start=1
    ):
        result["rank"] = rank
        final_results.append(result)

    return final_results
