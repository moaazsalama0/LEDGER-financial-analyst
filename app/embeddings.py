
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# Embedding Model
# ============================================================

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_embedding_model(
    model_name=DEFAULT_MODEL_NAME
):
    """
    Load the embedding model used by Retrieval.

    The model name is configurable so it can be changed
    later without modifying the retrieval logic.
    """

    return SentenceTransformer(model_name)


# ============================================================
# Prepare Texts
# ============================================================

def prepare_texts(chunks):
    """
    Extract valid textual content from retrieval chunks.

    Returns:
        texts       -> texts that will be embedded
        valid_chunks -> chunks corresponding to those texts

    Keeping the chunks alongside the texts is important because
    FAISS returns vector positions, and we need to map each
    vector back to its original chunk metadata.
    """

    texts = []
    valid_chunks = []

    for chunk in chunks:

        content = chunk.get("content", "")

        if content is None:
            continue

        content = str(content).strip()

        if not content:
            continue

        texts.append(content)
        valid_chunks.append(chunk)

    if not texts:
        raise ValueError(
            "No valid text found in chunks."
        )

    return texts, valid_chunks


# ============================================================
# Generate Embeddings
# ============================================================

def generate_embeddings(
    chunks,
    embedding_model,
    batch_size=32
):
    """
    Generate normalized embeddings for arbitrary chunks.

    Normalization allows cosine similarity to be computed
    efficiently using FAISS inner product.
    """

    texts, valid_chunks = prepare_texts(chunks)

    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    return embeddings, valid_chunks


# ============================================================
# Query Embedding
# ============================================================

def embed_query(
    query,
    embedding_model
):
    """
    Convert a search query into a normalized embedding.
    """

    if not query or not str(query).strip():
        raise ValueError(
            "Query must not be empty."
        )

    query_embedding = embedding_model.encode(
        [str(query)],
        normalize_embeddings=True
    )

    return np.asarray(
        query_embedding,
        dtype="float32"
    )
