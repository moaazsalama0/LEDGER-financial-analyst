
import re
import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str):

    text = text.lower()

    return re.findall(
        r"\b\w+\b",
        text
    )


class BM25Retriever:

    def __init__(self):

        self.index = None
        self.chunks = []

    def build(self, chunks):

        self.chunks = chunks

        tokenized_corpus = [
            tokenize(chunk["content"])
            for chunk in chunks
        ]

        if tokenized_corpus:

            self.index = BM25Okapi(
                tokenized_corpus
            )

        else:
            self.index = None

    def search(
        self,
        query: str,
        top_k: int = 30,
    ):

        if self.index is None:
            return []

        top_k = min(
            top_k,
            len(self.chunks)
        )

        query_tokens = tokenize(query)

        scores = self.index.get_scores(
            query_tokens
        )

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for rank, idx in enumerate(
            top_indices,
            start=1
        ):

            chunk = self.chunks[idx]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "rank": rank,
                "score": float(scores[idx]),
                "chunk": chunk,
            })

        return results
