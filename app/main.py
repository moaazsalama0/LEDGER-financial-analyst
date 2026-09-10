from typing import List, Optional, Any
import time
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .chunking import build_retrieval_corpus
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever
from .reranker import Reranker


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="LEDGER Retrieval API",
    version="1.0.0",
)


# ============================================================
# Document Normalization
# ============================================================

def normalize_document(doc):
    """
    Normalize a document received from the Document Processor
    into the structure expected by the Retrieval pipeline.
    """

    normalized = dict(doc)

    normalized.setdefault("sections", [])
    normalized.setdefault("pages", [])
    normalized.setdefault("tables", [])
    normalized.setdefault("processing", {})

    # --------------------------------------------------------
    # Document title
    # --------------------------------------------------------

    if not normalized.get("document_title"):
        normalized["document_title"] = normalized.get(
            "filename",
            normalized.get(
                "document_id",
                "unknown_document"
            )
        )

    normalized_pages = []

    # ========================================================
    # Normalize Pages
    # ========================================================

    for page in normalized["pages"]:

        page_copy = dict(page)

        page_copy.setdefault(
            "blocks",
            []
        )

        page_copy.setdefault(
            "reading_order",
            []
        )

        normalized_blocks = []

        # ====================================================
        # Normalize Blocks
        # ====================================================

        for block in page_copy["blocks"]:

            block_copy = dict(block)

            # ------------------------------------------------
            # Basic metadata
            # ------------------------------------------------

            block_copy.setdefault(
                "block_id",
                None
            )

            block_copy.setdefault(
                "page_number",
                page_copy.get("page_number")
            )

            block_copy.setdefault(
                "type",
                "text"
            )

            block_copy.setdefault(
                "heading_level",
                None
            )

            block_copy.setdefault(
                "section_id",
                None
            )

            block_copy.setdefault(
                "order_index",
                None
            )

            # ------------------------------------------------
            # Processor: text
            # Retrieval: content
            # ------------------------------------------------

            if not block_copy.get("content"):
                block_copy["content"] = block_copy.get(
                    "text",
                    ""
                )

            # ------------------------------------------------
            # Processor: bbox
            # Retrieval: bounding_box
            # ------------------------------------------------

            if not block_copy.get("bounding_box"):
                block_copy["bounding_box"] = block_copy.get(
                    "bbox"
                )

            # ------------------------------------------------
            # Content type
            # ------------------------------------------------

            if not block_copy.get("content_type"):

                if block_copy.get("type") == "table":
                    block_copy["content_type"] = "table"
                else:
                    block_copy["content_type"] = "text"

            # ------------------------------------------------
            # Resolve section title
            # ------------------------------------------------

            if not block_copy.get("section"):

                section_id = block_copy.get(
                    "section_id"
                )

                section_title = None

                if section_id:

                    for section in normalized.get(
                        "sections",
                        []
                    ):

                        if section.get(
                            "section_id"
                        ) == section_id:

                            section_title = section.get(
                                "title"
                            )

                            break

                block_copy["section"] = section_title

            # ------------------------------------------------
            # Table ID
            # ------------------------------------------------

            block_copy.setdefault(
                "table_id",
                None
            )

            normalized_blocks.append(
                block_copy
            )

        page_copy["blocks"] = normalized_blocks

        normalized_pages.append(
            page_copy
        )

    normalized["pages"] = normalized_pages

    return normalized


# ============================================================
# Pydantic Models
# ============================================================

class OCRBlock(BaseModel):

    block_id: Optional[str] = None

    page_number: Optional[int] = None

    type: Optional[str] = "text"

    heading_level: Optional[int] = None

    section_id: Optional[str] = None

    order_index: Optional[int] = None

    section: Optional[str] = None

    content: str = ""

    content_type: str = "text"

    bounding_box: Optional[Any] = None

    table_id: Optional[str] = None


class OCRPage(BaseModel):

    page_number: int

    blocks: List[OCRBlock]

    reading_order: List[Any] = []


class DocumentInput(BaseModel):

    document_id: str

    document_title: Optional[str] = None

    filename: Optional[str] = None

    pages: List[OCRPage]

    sections: List[Any] = []

    tables: List[Any] = []

    processing: dict = {}


class IngestRequest(BaseModel):

    documents: List[DocumentInput]


class SearchRequest(BaseModel):

    trace_id: str

    query: str

    top_k: int = Field(
        default=5,
        ge=1,
        le=30
    )


class RetrievedEvidence(BaseModel):

    document_id: str

    chunk_id: str

    rank: int

    score: float

    document_title: Optional[str] = None

    page_number: Optional[int] = None

    section: Optional[str] = None

    content: str

    content_type: str

    table_id: Optional[str] = None

    bounding_box: Optional[Any] = None


class SearchResponse(BaseModel):

    trace_id: str

    latency_ms: float

    evidence: List[RetrievedEvidence]


# ============================================================
# Documents Response Models
# ============================================================

class DocumentSummary(BaseModel):

    document_id: str

    document_title: Optional[str] = None

    filename: Optional[str] = None

    page_count: int

    table_count: int

    tables: List[Any] = []

    indexed_at: str


class DocumentsResponse(BaseModel):

    count: int

    documents: List[DocumentSummary]


# ============================================================
# Retrieval Components
# ============================================================

embedding_model = EmbeddingModel()

vector_store = VectorStore(
    embedding_model
)

bm25_retriever = BM25Retriever()

hybrid_retriever = HybridRetriever(
    vector_store=vector_store,
    bm25_retriever=bm25_retriever,
    rrf_k=60,
)

reranker = Reranker()


# ============================================================
# Retrieval State
# ============================================================

# All currently ingested documents.
#
# Example:
#
# DOCUMENTS = {
#     "doc_001": {...},
#     "doc_002": {...},
#     "doc_003": {...}
# }

DOCUMENTS = {}

# Combined chunks from all ingested documents.

RETRIEVAL_CORPUS = []


# ============================================================
# POST /ingest
# ============================================================

@app.post("/ingest")
def ingest(request: IngestRequest):

    global DOCUMENTS
    global RETRIEVAL_CORPUS

    # ========================================================
    # 1. Normalize and validate all incoming documents
    # ========================================================

    incoming_documents = []

    for raw_document in request.documents:

        # Convert Pydantic object back to a dictionary
        # containing the Processor-compatible structure.

        raw_dict = raw_document.model_dump()

        normalized_document = normalize_document(
            raw_dict
        )

        # Validate normalized document again.

        validated_document = DocumentInput.model_validate(
            normalized_document
        )

        incoming_documents.append(
            normalized_document
        )

    # ========================================================
    # 2. Add / Update all documents
    # ========================================================

    # Documents already stored remain available.
    #
    # If the same document_id is received again,
    # that document is updated/replaced.

    for document in incoming_documents:

        document_id = document.get(
            "document_id"
        )

        # Store indexing timestamp for the document.
        document["indexed_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        DOCUMENTS[document_id] = document

    # ========================================================
    # 3. Build the COMPLETE retrieval corpus
    # ========================================================

    RETRIEVAL_CORPUS = []

    document_chunk_counts = {}

    for document_id, stored_document in DOCUMENTS.items():

        document_chunks = build_retrieval_corpus(
            stored_document
        )

        RETRIEVAL_CORPUS.extend(
            document_chunks
        )

        document_chunk_counts[document_id] = len(
            document_chunks
        )

    # ========================================================
    # 4. Build Dense Vector Index ONCE
    # ========================================================

    vector_store.build(
        RETRIEVAL_CORPUS
    )

    # ========================================================
    # 5. Build BM25 Index ONCE
    # ========================================================

    bm25_retriever.build(
        RETRIEVAL_CORPUS
    )

    # ========================================================
    # 6. Debug Information
    # ========================================================

    print(
        "\n========== INGEST DEBUG =========="
    )

    print(
        "Documents received in request:",
        len(incoming_documents)
    )

    print(
        "Documents stored:",
        len(DOCUMENTS)
    )

    print(
        "Total chunks:",
        len(RETRIEVAL_CORPUS)
    )

    for document in incoming_documents:

        document_id = document.get(
            "document_id"
        )

        print(
            f"Document {document_id}: "
            f"{document_chunk_counts.get(document_id, 0)} chunks"
        )

    print(
        "==================================\n"
    )

    # ========================================================
    # 7. Response
    # ========================================================

    return {

        "status":
            "success",

        "documents_received":
            len(incoming_documents),

        "documents_stored":
            len(DOCUMENTS),

        "chunks_indexed":
            len(RETRIEVAL_CORPUS),

        "document_chunk_counts":
            {
                document_id:
                    document_chunk_counts.get(
                        document_id,
                        0
                    )

                for document_id in [
                    document.get("document_id")
                    for document in incoming_documents
                ]
            },
    }


# ============================================================
# GET /documents
# ============================================================

@app.get(
    "/documents",
    response_model=DocumentsResponse
)
def get_documents():

    documents = []

    for document in DOCUMENTS.values():

        pages = document.get(
            "pages",
            []
        )

        tables = document.get(
            "tables",
            []
        )

        documents.append(
            DocumentSummary(

                document_id=document.get(
                    "document_id"
                ),

                document_title=document.get(
                    "document_title"
                ),

                filename=document.get(
                    "filename",
                    document.get(
                        "document_title"
                    )
                ),

                page_count=len(pages),

                table_count=len(tables),

                tables=tables,

                indexed_at=document.get(
                    "indexed_at"
                ),
            )
        )

    return DocumentsResponse(

        count=len(documents),

        documents=documents,
    )


# ============================================================
# POST /search
# ============================================================

@app.post(
    "/search",
    response_model=SearchResponse
)
def search(request: SearchRequest):

    start_time = time.perf_counter()

    trace_id = request.trace_id

    query = request.query

    top_k = request.top_k

    # ========================================================
    # 1. Hybrid Retrieval
    # Dense + BM25
    # ========================================================

    rrf_results = hybrid_retriever.search(

        query=query,

        retrieval_k=30,

        final_k=30,
    )

    # ========================================================
    # 2. Reranking
    # ========================================================

    reranked_results = reranker.rerank(

        query=query,

        candidates=rrf_results,

        top_k=top_k,
    )

    # ========================================================
    # 3. Build Retrieved Evidence
    # ========================================================

    evidence = []

    for result in reranked_results:

        chunk = result["chunk"]

        evidence.append(

            RetrievedEvidence(

                document_id=
                    chunk["document_id"],

                chunk_id=
                    result["chunk_id"],

                rank=
                    result["rank"],

                score=
                    result["reranker_score"],

                document_title=
                    chunk.get(
                        "document_title"
                    ),

                page_number=
                    chunk.get(
                        "page_number"
                    ),

                section=
                    chunk.get(
                        "section"
                    ),

                content=
                    chunk["content"],

                content_type=
                    chunk.get(
                        "content_type",
                        "text"
                    ),

                table_id=
                    chunk.get(
                        "table_id"
                    ),

                bounding_box=
                    chunk.get(
                        "bounding_box"
                    ),
            )
        )

    # ========================================================
    # 4. Calculate Latency
    # ========================================================

    latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    # ========================================================
    # 5. Return Search Response
    # ========================================================

    return {

        "trace_id":
            trace_id,

        "latency_ms":
            round(
                latency_ms,
                2
            ),

        "evidence":
            evidence,
    }


# ============================================================
# GET /health
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "ok",

        "documents":
            len(DOCUMENTS),

        "chunks":
            len(RETRIEVAL_CORPUS),
    }
