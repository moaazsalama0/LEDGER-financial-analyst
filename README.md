# LEDGER — Retrieval API

The **Retrieval API** is the retrieval component of the **LEDGER — Financial Document Intelligence Agent** system.

Its main responsibility is to transform OCR-processed financial documents into searchable retrieval units and return **relevant, ranked, and traceable evidence** for the Agent.

The service combines:

* Dense semantic retrieval
* BM25 lexical retrieval
* Reciprocal Rank Fusion (RRF)
* Reranking
* Top-K evidence selection

The Retrieval API **does not generate the final answer**. It provides the evidence that the Agent uses for reasoning and answer generation.

---

## 1. Role in the LEDGER System

```text
                    User
                      │
                      ▼
                Orchestrator
                      │
                      ▼
                    Agent
                      │
              needs evidence
                      │
                      ▼
               POST /search
                      │
                      ▼
              Retrieval API
                      │
          Dense + BM25 Retrieval
                      │
                     RRF
                      │
                  Reranker
                      │
                      ▼
                Top-K Evidence
                      │
                      ▼
                    Agent
                      │
              Generate Answer
                      │
                      ▼
                  Validator
                      │
                      ▼
                Orchestrator
                      │
                      ▼
                  Final Response
```

The Retrieval API is therefore an **evidence retrieval service**, while the Agent is responsible for reasoning over that evidence.

---

## 2. Responsibilities

The Retrieval API is responsible for:

1. Receiving OCR-processed documents for indexing.
2. Normalizing document and OCR metadata.
3. Preserving document traceability information.
4. Splitting document content into retrieval chunks.
5. Generating dense embeddings.
6. Building the vector index.
7. Building the BM25 index.
8. Performing hybrid retrieval.
9. Combining dense and lexical results using RRF.
10. Reranking retrieved candidates.
11. Returning the most relevant Top-K evidence.
12. Preserving metadata such as page number, section, table ID, and bounding box.
13. Returning `trace_id` and retrieval latency for request tracing and evaluation.

---

# 3. Retrieval Pipeline

The internal retrieval pipeline is:

```text
OCR Document
     │
     ▼
Normalization
     │
     ▼
Chunking
     │
     ├──────────────► Dense Embeddings
     │                       │
     │                       ▼
     │                  Vector Index
     │
     └──────────────► BM25 Index
                              │
                              ▼
                        Hybrid Retrieval
                              │
                              ▼
                             RRF
                              │
                              ▼
                        Candidate Results
                              │
                              ▼
                           Reranker
                              │
                              ▼
                          Top-K Evidence
```

The indexing stages are performed when a document is ingested.

The retrieval stages are performed when the Agent sends a search request.

---

# 4. Integration with the Orchestrator

The Retrieval API interacts with the rest of the system through HTTP API contracts.

The two main interactions are:

```text
Orchestrator → POST /ingest → Retrieval API
```

and:

```text
Agent → POST /search → Retrieval API
```

These two endpoints serve different purposes.

### `/ingest`

Used during **document ingestion and indexing**.

The Orchestrator sends the OCR output to the Retrieval API after Document Processing/OCR has completed.

### `/search`

Used during **question answering**.

The Agent calls the Retrieval API when it needs relevant evidence to answer the user's question.

---

# 5. `/ingest` — Document Indexing

## Who calls `/ingest`?

The **Orchestrator** calls `/ingest` as part of the document ingestion workflow.

```text
Document
   ↓
Document Processing
   ↓
OCR
   ↓
OCR JSON
   ↓
Orchestrator
   ↓
POST /ingest
   ↓
Retrieval API
   ↓
Normalization
   ↓
Chunking
   ↓
Dense Index + BM25 Index
```

The purpose of `/ingest` is to prepare the document so that it can later be searched.

### Important

`/ingest` is an **indexing operation**.

It is **not called for every user question**.

A document should first be ingested and indexed. Once the document is indexed, the Agent can perform multiple `/search` requests against it.

### Example response

```json
{
  "status": "success",
  "document_id": "doc_001",
  "document_title": "financial_report.pdf",
  "chunks_indexed": 24
}
```

---

# 6. `/search` — Evidence Retrieval

## Who calls `/search`?

The **Agent** calls `/search`.

The Agent is responsible for deciding when retrieval is needed during the question-answering process.

```text
User Question
      ↓
Orchestrator
      ↓
Agent
      ↓
Agent needs evidence
      ↓
POST /search
      ↓
Retrieval API
```

The Retrieval API receives the query and returns ranked evidence.

### Example request

```json
{
  "trace_id": "trace_001",
  "query": "What was the decrease in the share of net earnings in Golar Partners?",
  "top_k": 5
}
```

---

# 7. What Happens Inside `/search`?

Once the Agent sends a search request, the Retrieval API performs the following pipeline:

```text
Query
  │
  ├──────────────► Dense Retrieval
  │
  └──────────────► BM25 Retrieval
                         │
                         ▼
                  Reciprocal Rank Fusion
                         │
                         ▼
                    Candidates
                         │
                         ▼
                      Reranker
                         │
                         ▼
                     Top-K Results
```

### Step 1 — Dense Retrieval

The query is converted into an embedding and compared with the indexed document chunks.

This captures **semantic similarity**, allowing the system to retrieve content even when the exact query wording does not appear in the document.

### Step 2 — BM25 Retrieval

BM25 performs lexical retrieval based on the terms appearing in the query and document chunks.

This is useful for exact terminology, names, financial terms, and specific values.

### Step 3 — Reciprocal Rank Fusion

The dense and BM25 rankings are combined using **RRF**.

This produces a unified candidate ranking from both retrieval methods.

### Step 4 — Reranking

The candidate results are passed to the reranker.

The reranker evaluates the relevance between the query and each candidate chunk more precisely.

### Step 5 — Top-K Selection

The highest-ranked results are returned as evidence.

The requested `top_k` determines how many evidence items are returned.

---

# 8. `/search` Response

The response contains the original `trace_id`, retrieval latency, and ranked evidence.

```json
{
  "trace_id": "trace_001",
  "latency_ms": 257.36,
  "evidence": [
    {
      "document_id": "doc_001",
      "chunk_id": "doc_001_p1_b001_c0",
      "rank": 1,
      "score": 0.9886,
      "document_title": "financial_report.pdf",
      "page_number": 1,
      "section": "Equity in net earnings of affiliates",
      "content": "Relevant financial document content...",
      "content_type": "text",
      "table_id": null,
      "bounding_box": {
        "x0": 49.5,
        "y0": 162.31,
        "x1": 562.5,
        "y1": 218.54
      }
    }
  ]
}
```

---

# 9. How the Agent Uses the Search Response

The Agent receives the retrieved evidence and uses it as the grounding context for reasoning.

```text
User Question
      │
      ▼
    Agent
      │
      │ POST /search
      ▼
Retrieval API
      │
      │ Ranked Evidence
      ▼
    Agent
      │
      │ Question + Evidence
      ▼
Generate Answer
      │
      ▼
  Validator
```

The Agent should base its answer on the retrieved evidence rather than relying on unsupported information.

The Retrieval API itself does not generate or modify the final answer.

---

# 10. Evidence Contract

Each retrieved evidence item contains:

| Field            | Description                                     |
| ---------------- | ----------------------------------------------- |
| `document_id`    | Unique identifier of the source document        |
| `chunk_id`       | Unique identifier of the retrieved chunk        |
| `rank`           | Final retrieval rank                            |
| `score`          | Final reranker score                            |
| `document_title` | Source document title                           |
| `page_number`    | Page containing the evidence                    |
| `section`        | Section or heading associated with the evidence |
| `content`        | Retrieved text/content                          |
| `content_type`   | `text` or `table`                               |
| `table_id`       | Table identifier when applicable                |
| `bounding_box`   | Original location of the evidence on the page   |

---

# 11. Document Metadata and Traceability

Traceability is an important requirement for financial document retrieval.

The Retrieval API preserves metadata from the OCR stage so that every retrieved chunk can be traced back to its source.

The main traceability fields are:

```text
document_id
    ↓
chunk_id
    ↓
page_number
    ↓
section
    ↓
bounding_box
```

This allows downstream components to identify **where the evidence came from in the original document**.

The `bounding_box` can also be used later for document visualization or highlighting the retrieved content on the original page.

---

# 12. Retrieval Configuration

The current search configuration includes:

```text
Retrieval candidates: 30
Final results: top_k
RRF k: 60
Default top_k: 5
Maximum top_k: 30
```

The `/search` endpoint accepts a configurable `top_k` value.

---

# 13. API Summary

| Method | Endpoint  | Caller                   | Purpose                                 |
| ------ | --------- | ------------------------ | --------------------------------------- |
| `POST` | `/ingest` | Orchestrator             | Normalize, chunk, and index a document  |
| `POST` | `/search` | Agent                    | Retrieve ranked evidence for a question |
| `GET`  | `/health` | Any service / monitoring | Check service status                    |

---

# 14. End-to-End Integration Flow

The complete LEDGER flow is:

```text
                    ┌──────────────┐
                    │     User     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Orchestrator │
                    └──────┬───────┘
                           │
                  ┌────────┴─────────┐
                  │                  │
                  ▼                  ▼
          Document Processing      Agent
                  │                  │
                  ▼                  │
                 OCR                 │
                  │                  │
                  ▼                  │
          POST /ingest               │
                  │                  │
                  ▼                  │
          Retrieval API              │
                  │                  │
          Index Document             │
                                     │
                              needs evidence
                                     │
                                     ▼
                              POST /search
                                     │
                                     ▼
                              Retrieval API
                                     │
                          Dense + BM25 + RRF
                                     │
                                  Reranker
                                     │
                                     ▼
                                Evidence
                                     │
                                     ▼
                                   Agent
                                     │
                              Generate Answer
                                     │
                                     ▼
                                 Validator
                                     │
                                     ▼
                               Orchestrator
                                     │
                                     ▼
                                  User
```

---

# 15. Important Integration Rule

The service ownership for Retrieval communication is:

```text
Orchestrator → POST /ingest
Agent        → POST /search
```

The Orchestrator should **not** directly access the internal retrieval components.

The following components are internal to the Retrieval API:

```text
EmbeddingModel
VectorStore
BM25Retriever
HybridRetriever
Reranker
```

Other services communicate with Retrieval through its public HTTP endpoints.

This keeps the Retrieval API modular and allows its internal implementation to change without requiring changes in the Agent or Orchestrator.

---




