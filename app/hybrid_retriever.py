
class HybridRetriever:

    def __init__(
        self,
        vector_store,
        bm25_retriever,
        rrf_k: int = 60,
    ):

        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def reciprocal_rank_fusion(
        self,
        dense_results,
        bm25_results,
        top_k=30,
    ):

        rrf_scores = {}
        chunk_lookup = {}

        # -----------------------------
        # Dense contribution
        # -----------------------------

        for result in dense_results:

            chunk_id = result["chunk_id"]

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + 1.0 / (
                    self.rrf_k
                    + result["rank"]
                )
            )

            chunk_lookup[chunk_id] = (
                result["chunk"]
            )

        # -----------------------------
        # BM25 contribution
        # -----------------------------

        for result in bm25_results:

            chunk_id = result["chunk_id"]

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + 1.0 / (
                    self.rrf_k
                    + result["rank"]
                )
            )

            chunk_lookup[chunk_id] = (
                result["chunk"]
            )

        # -----------------------------
        # Sort by RRF score
        # -----------------------------

        ranked = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []

        for rank, (
            chunk_id,
            score,
        ) in enumerate(
            ranked[:top_k],
            start=1,
        ):

            results.append({
                "chunk_id": chunk_id,
                "rank": rank,
                "rrf_score": float(score),
                "chunk": chunk_lookup[chunk_id],
            })

        return results

    def search(
        self,
        query: str,
        retrieval_k: int = 200,
        final_k: int = 30,
    ):

        # Dense Top 200
        dense_results = (
            self.vector_store.search(
                query,
                top_k=retrieval_k,
            )
        )

        # BM25 Top 200
        bm25_results = (
            self.bm25_retriever.search(
                query,
                top_k=retrieval_k,
            )
        )

        # RRF → Top 30
        return self.reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            top_k=final_k,
        )
