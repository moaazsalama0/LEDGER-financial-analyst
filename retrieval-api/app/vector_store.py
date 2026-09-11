
import faiss
import numpy as np


class VectorStore:

    def __init__(self, embedding_model):

        self.embedding_model = embedding_model

        self.index = None
        self.chunks = []

    def build(self, chunks):

        self.chunks = chunks

        if not chunks:
            self.index = None
            return

        texts = [
            chunk["content"]
            for chunk in chunks
        ]

        embeddings = (
            self.embedding_model
            .encode_documents(texts)
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 30,
    ):

        if self.index is None:
            return []

        if not self.chunks:
            return []

        top_k = min(
            top_k,
            len(self.chunks)
        )

        query_embedding = (
            self.embedding_model
            .encode_query(query)
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):

            if idx < 0:
                continue

            chunk = self.chunks[idx]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "rank": rank,
                "score": float(score),
                "chunk": chunk,
            })

        return results
