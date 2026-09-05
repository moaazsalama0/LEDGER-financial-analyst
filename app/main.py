from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
import time


# =========================
# Chunking
# =========================

from app.chunking import (
    normalize_blocks,
    section_aware_chunking,
    table_aware_chunking
)


# =========================
# Embeddings
# =========================

from app.embeddings import (
    load_embedding_model,
    generate_embeddings,
    embed_query
)


# =========================
# Vector Store
# =========================

from app.vector_store import (
    build_vector_index,
    build_metadata_store,
    validate_index_metadata,
    dense_search
)


# =========================
# BM25
# =========================

from app.bm25_retriever import (
    build_bm25_index,
    bm25_search
)


# =========================
# Hybrid Retrieval
# =========================

from app.hybrid_retriever import (
    reciprocal_rank_fusion
)


# =========================
# Reranker
# =========================

from app.reranker import (
    load_reranker,
    rerank_results
)


# =========================
# FastAPI App
# =========================

app = FastAPI(
    title="LEDGER Retrieval API",
    version="1.0.0"
)


# =========================
# Configuration
# =========================

CHUNK_SIZE = 1000
EMBEDDING_BATCH_SIZE = 32

RETRIEVAL_K = 30
FINAL_K = 5


# =========================
# Request Schemas
# =========================

class IngestRequest(BaseModel):
    document_id: str
    document_title: str | None = None
    pages: List[Dict[str, Any]]
    sections: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []


class SearchRequest(BaseModel):
    trace_id: str | None = None
    query: str
    top_k: int = RETRIEVAL_K


# =========================
# Runtime State
# =========================

embedding_model = load_embedding_model()
reranker = load_reranker()

vector_index = None
vector_metadata = []

bm25_index = None
bm25_chunks = []

retrieval_chunks = []


# =========================
# Evidence Helpers
# =========================

def get_evidence_page_number(page_number):

    if isinstance(page_number, list):

        if not page_number:
            return None

        return page_number[0]

    return page_number


# =========================
# Health Check
# =========================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "retrieval-api"
    }


# =========================
# Ingest
# =========================

@app.post("/ingest")
def ingest_document(request: IngestRequest):

    global vector_index
    global vector_metadata
    global bm25_index
    global bm25_chunks
    global retrieval_chunks

    try:

        # Convert request to dictionary
        doc = request.model_dump()

        # -------------------------
        # Normalize OCR blocks
        # -------------------------

        normalized_blocks = normalize_blocks(doc)

        # -------------------------
        # Section-Aware text chunks
        # -------------------------

        text_chunks = section_aware_chunking(
            normalized_blocks,
            chunk_size=CHUNK_SIZE
        )

        # -------------------------
        # Table-Aware chunks
        # -------------------------

        table_chunks = table_aware_chunking(doc)

        # -------------------------
        # Combine Retrieval Corpus
        # -------------------------

        retrieval_chunks = (
            text_chunks +
            table_chunks
        )

        if not retrieval_chunks:
            raise ValueError(
                "No valid chunks were created from the document."
            )

        # -------------------------
        # Dense Embeddings
        # -------------------------

        embeddings, valid_chunks = generate_embeddings(
            retrieval_chunks,
            embedding_model,
            batch_size=EMBEDDING_BATCH_SIZE
        )

        # -------------------------
        # FAISS Index
        # -------------------------

        vector_index = build_vector_index(
            embeddings
        )

        vector_metadata = build_metadata_store(
            valid_chunks
        )

        validate_index_metadata(
            vector_index,
            vector_metadata
        )

        # -------------------------
        # BM25 Index
        # -------------------------

        bm25_index, bm25_chunks = build_bm25_index(
            valid_chunks
        )

        return {
            "status": "success",
            "document_id": request.document_id,
            "num_text_chunks": len(text_chunks),
            "num_table_chunks": len(table_chunks),
            "num_retrieval_chunks": len(retrieval_chunks),
            "num_vectors": vector_index.ntotal
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================
# Search
# =========================

@app.post("/search")
def search(request: SearchRequest):

    global vector_index
    global vector_metadata
    global bm25_index
    global bm25_chunks

    start_time = time.perf_counter()

    try:

        if vector_index is None or bm25_index is None:
            raise ValueError(
                "No document has been ingested yet."
            )

        if not request.query.strip():
            raise ValueError(
                "Query must not be empty."
            )

        # -------------------------
        # Dense Retrieval
        # -------------------------

        query_embedding = embed_query(
            request.query,
            embedding_model
        )

        dense_results = dense_search(
            query_embedding,
            vector_index,
            vector_metadata,
            top_k=RETRIEVAL_K
        )

        # -------------------------
        # BM25 Retrieval
        # -------------------------

        bm25_results = bm25_search(
            request.query,
            bm25_index,
            bm25_chunks,
            top_k=RETRIEVAL_K
        )

        # -------------------------
        # Hybrid Retrieval - RRF
        # -------------------------

        rrf_results = reciprocal_rank_fusion(
            [dense_results, bm25_results],
            top_k=RETRIEVAL_K
        )

        # -------------------------
        # Reranking
        # -------------------------

        reranked_results = rerank_results(
            request.query,
            rrf_results,
            reranker,
            top_k=FINAL_K
        )

        # -------------------------
        # Final Evidence
        # -------------------------

        evidence = []

        for result in reranked_results:

            evidence.append({
                "document_id": result["document_id"],
                "chunk_id": result["chunk_id"],
                "rank": result["rank"],
                "score": result.get(
                    "reranker_score",
                    result.get("rrf_score", 0.0)
                ),
                "document_title": result["document_title"],
                "page_number": get_evidence_page_number(
                    result["page_number"]
                ),
                "section": result["section"],
                "content": result["content"],
                "content_type": result["content_type"],
                "table_id": result["table_id"],
                "bounding_box": result["bounding_box"]
            })

        # -------------------------
        # Latency
        # -------------------------

        latency = time.perf_counter() - start_time

        return {
            "trace_id": request.trace_id,
            "latency": latency,
            "evidence": evidence
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

