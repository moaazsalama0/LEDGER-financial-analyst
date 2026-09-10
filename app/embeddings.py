%%writefile /content/retrieval-api/app/embeddings.py

from sentence_transformers import SentenceTransformer
import numpy as np


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:

    def __init__(self, model_name: str = DEFAULT_MODEL):

        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )

    def encode_documents(self, texts):

        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.astype("float32")

    def encode_query(self, query):

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding.astype("float32")
