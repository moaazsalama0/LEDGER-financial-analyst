
import faiss
import numpy as np


# ============================================================
# Build Vector Index
# ============================================================

def build_vector_index(embeddings):
    """
    Build a FAISS inner-product index from normalized embeddings.

    Since embeddings are normalized, inner product is equivalent
    to cosine similarity.
    """

    if embeddings is None:
        raise ValueError("Embeddings must not be None.")

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D array."
        )

    if len(embeddings) == 0:
        raise ValueError(
            "Cannot build an index from empty embeddings."
        )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ============================================================
# Build Metadata Store
# ============================================================

def build_metadata_store(chunks):
    """
    Keep the metadata associated with every indexed chunk.

    FAISS stores vectors, not document metadata.
    The metadata store allows us to map a returned FAISS
    vector position back to the original retrieval chunk.
    """

    metadata_store = []

    for chunk in chunks:

        metadata_store.append({
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "document_title": chunk["document_title"],
            "page_number": chunk["page_number"],
            "section": chunk["section"],
            "content": chunk["content"],
            "content_type": chunk["content_type"],
            "table_id": chunk["table_id"],
            "bounding_box": chunk["bounding_box"]
        })

    return metadata_store


# ============================================================
# Validate Index / Metadata Alignment
# ============================================================

def validate_index_metadata(
    index,
    metadata_store
):
    """
    Make sure every FAISS vector has exactly one
    corresponding metadata record.
    """

    if index.ntotal != len(metadata_store):

        raise ValueError(
            "FAISS index and metadata store are misaligned: "
            f"{index.ntotal} vectors vs "
            f"{len(metadata_store)} metadata records."
        )

    return True


# ============================================================
# Dense Vector Search
# ============================================================

def dense_search(
    query_embedding,
    index,
    metadata_store,
    top_k=5
):
    """
    Search the FAISS index using a query embedding.

    Returns evidence objects following the Retrieval
    response structure.
    """

    if index is None:
        raise ValueError(
            "Vector index must not be None."
        )

    if not metadata_store:
        return []

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    if query_embedding.ndim != 2:
        raise ValueError(
            "Query embedding must be a 2D array."
        )

    if query_embedding.shape[1] != index.d:
        raise ValueError(
            "Query embedding dimension does not match "
            "the vector index dimension."
        )

    k = min(
        max(int(top_k), 1),
        index.ntotal
    )

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx < 0:
            continue

        result = metadata_store[idx].copy()

        result["rank"] = rank
        result["score"] = float(score)

        results.append(result)

    return results
