
# ============================================================
# Hybrid Retrieval using Reciprocal Rank Fusion (RRF)
# ============================================================


def reciprocal_rank_fusion(
    result_lists,
    top_k=5,
    rrf_k=60
):
    """
    Combine ranked results from multiple retrieval methods
    using Reciprocal Rank Fusion (RRF).

    RRF formula:

        RRF(d) = sum(1 / (rrf_k + rank))

    Parameters
    ----------
    result_lists : list[list[dict]]
        Ranked results from different retrieval methods.

    top_k : int
        Number of final fused results.

    rrf_k : int
        RRF constant. 60 is a common default.

    Returns
    -------
    list[dict]
        Combined and reranked evidence.
    """

    if not result_lists:
        return []

    if top_k <= 0:
        return []

    rrf_scores = {}
    result_lookup = {}

    # --------------------------------------------------------
    # Process every retrieval result list
    # --------------------------------------------------------

    for results in result_lists:

        if not results:
            continue

        for rank, result in enumerate(results, start=1):

            chunk_id = result.get("chunk_id")

            if not chunk_id:
                continue

            # RRF contribution from this retrieval method
            contribution = 1.0 / (
                rrf_k + rank
            )

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + contribution
            )

            # Keep the original chunk/evidence metadata
            result_lookup[chunk_id] = result

    if not rrf_scores:
        return []

    # --------------------------------------------------------
    # Sort by fused RRF score
    # --------------------------------------------------------

    ranked_chunks = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # --------------------------------------------------------
    # Build final results
    # --------------------------------------------------------

    final_results = []

    for rank, (chunk_id, rrf_score) in enumerate(
        ranked_chunks[:top_k],
        start=1
    ):

        result = result_lookup[chunk_id].copy()

        result["rank"] = rank
        result["rrf_score"] = float(
            rrf_score
        )

        final_results.append(result)

    return final_results
