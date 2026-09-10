%%writefile /content/retrieval-api/app/main.py

from typing import List, Optional, Any
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field


from .chunking import build_retrieval_corpus
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever
from .reranker import Reranker


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="LEDGER Retrieval API",
    version="1.0.0",
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_document(doc):
    """
    Convert raw OCR document schema into the schema expected
    by the Retrieval pipeline.
    """

    # Preserve the complete original OCR document
    normalized = dict(doc)

    # --------------------------------------------------------
    # Document-level fields
    # --------------------------------------------------------

    normalized.setdefault("sections", [])
    normalized.setdefault("pages", [])
    normalized.setdefault("tables", [])
    normalized.setdefault("processing", {})

    # OCR uses filename.
    # Retrieval uses document_title.
    if not normalized.get("document_title"):

        normalized["document_title"] = normalized.get(
            "filename",
            normalized.get(
                "document_id",
                "unknown_document"
            )
        )

    # --------------------------------------------------------
    # Pages
    # --------------------------------------------------------

    normalized_pages = []

    for page in normalized["pages"]:

        # Preserve all original page fields
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

        # ----------------------------------------------------
        # Blocks
        # ----------------------------------------------------

        for block in page_copy["blocks"]:

            # Preserve all original OCR fields
            block_copy = dict(block)

            # ------------------------------------------------
            # Original OCR metadata
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
            # text -> content
            # ------------------------------------------------

            if not block_copy.get("content"):

                block_copy["content"] = block_copy.get(
                    "text",
                    ""
                )

            # ------------------------------------------------
            # bbox -> bounding_box
            # ------------------------------------------------

            if not block_copy.get("bounding_box"):

                block_copy["bounding_box"] = block_copy.get(
                    "bbox"
                )

            # ------------------------------------------------
            # type -> content_type
            # ------------------------------------------------

            if not block_copy.get("content_type"):

                if block_copy.get("type") == "table":

                    block_copy["content_type"] = "table"

                else:

                    block_copy["content_type"] = "text"

            # ------------------------------------------------
            # Section
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
# INGESTION CONTRACT
# ============================================================

class OCRBlock(BaseModel):

    # Original OCR metadata
    block_id: Optional[str] = None

    page_number: Optional[int] = None

    type: Optional[str] = "text"

    heading_level: Optional[int] = None

    section_id: Optional[str] = None

    order_index: Optional[int] = None

    # Retrieval fields
    section: Optional[str] = None

    content: str = ""

    content_type: str = "text"

    bounding_box: Optional[Any] = None

    table_id: Optional[str] = None


class OCRPage(BaseModel):

    page_number: int

    blocks: List[OCRBlock]

    reading_order: List[Any] = []


class IngestRequest(BaseModel):

    document_id: str

    document_title: str

    pages: List[OCRPage]

    # Document-level fields required by the
    # retrieval chunking pipeline.
    sections: List[Any] = []

    tables: List[Any] = []

    processing: dict = {}


# ============================================================
# SEARCH CONTRACT
# ============================================================

class SearchRequest(BaseModel):

    trace_id: str

    query: str

    top_k: int = Field(
        default=5,
        ge=1,
        le=30,
    )


# ============================================================
# RETRIEVED EVIDENCE CONTRACT
# ============================================================

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
# RETRIEVAL COMPONENTS
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
# INTERNAL STATE
# ============================================================

DOCUMENTS = []

RETRIEVAL_CORPUS = []


# ============================================================
# DOCUMENT INGESTION
# ============================================================

@app.post("/ingest")
def ingest(raw_document: dict):

    global DOCUMENTS
    global RETRIEVAL_CORPUS

    # --------------------------------------------------------
    # Step 1: Normalize OCR output
    # --------------------------------------------------------

    normalized_document = normalize_document(
        raw_document
    )

    # --------------------------------------------------------
    # Step 2: Validate retrieval document
    # --------------------------------------------------------

    request = IngestRequest.model_validate(
        normalized_document
    )

    # --------------------------------------------------------
    # Step 3: Keep the ORIGINAL normalized document
    #
    # We validate the document with Pydantic, but we do NOT
    # use request.model_dump() here.
    #
    # This preserves all OCR fields that may be required by
    # the chunking pipeline.
    # --------------------------------------------------------

    document = normalized_document

    # --------------------------------------------------------
    # Store document
    # --------------------------------------------------------

    DOCUMENTS = [document]

    # --------------------------------------------------------
    # Step 4: Build retrieval corpus
    # --------------------------------------------------------

    RETRIEVAL_CORPUS = build_retrieval_corpus(
        document
    )

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    print("\n========== INGEST DEBUG ==========")

    print(
        "Document ID:",
        document.get("document_id")
    )

    print(
        "Document title:",
        document.get("document_title")
    )

    print(
        "Pages:",
        len(
            document.get(
                "pages",
                []
            )
        )
    )

    print(
        "Blocks:",
        sum(
            len(
                page.get(
                    "blocks",
                    []
                )
            )
            for page in document.get(
                "pages",
                []
            )
        )
    )

    print(
        "Sections:",
        len(
            document.get(
                "sections",
                []
            )
        )
    )

    print(
        "Tables:",
        len(
            document.get(
                "tables",
                []
            )
        )
    )

    print(
        "Chunks generated:",
        len(RETRIEVAL_CORPUS)
    )

    if RETRIEVAL_CORPUS:

        print(
            "First chunk:"
        )

        print(
            RETRIEVAL_CORPUS[0]
        )

    print("==================================\n")

    # --------------------------------------------------------
    # Step 5: Dense index
    # --------------------------------------------------------

    vector_store.build(
        RETRIEVAL_CORPUS
    )

    # --------------------------------------------------------
    # Step 6: BM25 index
    # --------------------------------------------------------

    bm25_retriever.build(
        RETRIEVAL_CORPUS
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "status": "success",

        "document_id": request.document_id,

        "document_title": request.document_title,

        "chunks_indexed": len(
            RETRIEVAL_CORPUS
        ),
    }


# ============================================================
# SEMANTIC EVIDENCE SEARCH
# ============================================================

@app.post(
    "/search",
    response_model=SearchResponse,
)
def search(
    request: SearchRequest
):

    # --------------------------------------------------------
    # Start latency measurement
    # --------------------------------------------------------

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Request parameters
    # --------------------------------------------------------

    trace_id = request.trace_id

    query = request.query

    top_k = request.top_k

    # --------------------------------------------------------
    # Hybrid retrieval
    # --------------------------------------------------------

    rrf_results = hybrid_retriever.search(
        query=query,
        retrieval_k=30,
        final_k=30,
    )

    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    reranked_results = reranker.rerank(
        query=query,
        candidates=rrf_results,
        top_k=top_k,
    )

    # --------------------------------------------------------
    # Build evidence
    # --------------------------------------------------------

    evidence = []

    for result in reranked_results:

        chunk = result["chunk"]

        evidence.append(
            RetrievedEvidence(

                document_id=chunk[
                    "document_id"
                ],

                chunk_id=result[
                    "chunk_id"
                ],

                rank=result[
                    "rank"
                ],

                score=result[
                    "reranker_score"
                ],

                document_title=chunk.get(
                    "document_title"
                ),

                page_number=chunk.get(
                    "page_number"
                ),

                section=chunk.get(
                    "section"
                ),

                content=chunk[
                    "content"
                ],

                content_type=chunk.get(
                    "content_type",
                    "text",
                ),

                table_id=chunk.get(
                    "table_id"
                ),

                bounding_box=chunk.get(
                    "bounding_box"
                ),
            )
        )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "trace_id": trace_id,

        "latency_ms": round(
            latency_ms,
            2
        ),

        "evidence": evidence,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",

        "documents": len(
            DOCUMENTS
        ),

        "chunks": len(
            RETRIEVAL_CORPUS
        ),
    }
