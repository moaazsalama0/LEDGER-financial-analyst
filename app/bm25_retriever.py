
import re
import numpy as np
from rank_bm25 import BM25Okapi


# ============================================================
# Tokenization
# ============================================================

def tokenize(text):
    """
    Basic tokenizer for BM25.

    Converts text to lowercase and extracts word-like tokens.
    """

    if text is None:
        return []

    return re.findall(
        r"\b\w+\b",
        str(text).lower()
    )


# ============================================================
# Build BM25 Index
# ============================================================

def build_bm25_index(chunks):
    """
    Build a BM25 index over arbitrary retrieval chunks.

    The index keeps the same ordering as the supplied chunks,
    which allows BM25 result positions to map back to chunks.
    """

    if not chunks:
        raise ValueError(
            "Cannot build BM25 index from empty chunks."
        )

    valid_chunks = []

    for chunk in chunks:

        content = chunk.get("content", "")

        if content is None:
            continue

        content = str(content).strip()

        if not content:
            continue

        valid_chunks.append(chunk)

    if not valid_chunks:
        raise ValueError(
            "No valid text found in chunks."
        )

    tokenized_corpus = [
        tokenize(chunk["content"])
        for chunk in valid_chunks
    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    return bm25, valid_chunks


# ============================================================
# BM25 Search
# ============================================================

def bm25_search(
    query,
    bm25_index,
    chunks,
    top_k=5
):
    """
    Search the BM25 index and return ranked evidence.

    Scores are BM25 scores and should primarily be used
    for ranking within this retrieval method.
    """

    if not query or not str(query).strip():
        raise ValueError(
            "Query must not be empty."
        )

    if bm25_index is None:
        raise ValueError(
            "BM25 index must not be None."
        )

    if not chunks:
        return []

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = bm25_index.get_scores(
        query_tokens
    )

    ranked_indices = np.argsort(
        scores
    )[::-1]

    k = min(
        max(int(top_k), 1),
        len(chunks)
    )

    results = []

    for rank, idx in enumerate(
        ranked_indices[:k],
        start=1
    ):

        result = chunks[idx].copy()

        result["rank"] = rank
        result["score"] = float(
            scores[idx]
        )

        results.append(result)

    return results
