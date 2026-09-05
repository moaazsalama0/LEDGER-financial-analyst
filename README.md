
# LEDGER Retrieval API

Retrieval service for the LEDGER financial document intelligence system.

## What Has Been Implemented

The Retrieval pipeline currently includes:

* OCR output normalization
* Section-aware text chunking
* Table-aware chunking
* Parent-child chunking as an experiment
* Dense embeddings using Sentence Transformers
* FAISS vector retrieval
* BM25 lexical retrieval
* Hybrid retrieval using RRF
* Cross-encoder reranking
* Retrieval evaluation metrics:

  * Recall@1
  * Recall@3
  * Recall@5
  * MRR
* FastAPI service with:

  * `GET /health`
  * `POST /ingest`
  * `POST /search`

Current pipeline:

```text
OCR
 ↓
Normalization
 ↓
Section-Aware + Table-Aware Chunking
 ↓
Dense Retrieval + BM25
 ↓
RRF
 ↓
Reranking
 ↓
Top 5 Evidence
```

The implementation has been tested end-to-end using mock structured OCR data.

---

## How the Next Developer Should Use It

The main integration point is `app/main.py`.

The Agent should communicate with Retrieval through the API rather than using the internal modules directly.

For a user question, the Agent sends:

```json
{
    "trace_id": "trace_123",
    "query": "What was the Net Sales in fiscal 2019?",
    "top_k": 5
}
```

to:

```text
POST /search
```

Retrieval returns ranked evidence containing:

```text
document_id
chunk_id
rank
score
document_title
page_number
section
content
content_type
table_id
bounding_box
```

The Agent then uses the returned `content` and metadata as evidence for answering the question.

For calculation questions, Retrieval only returns the required evidence. The Agent is responsible for the reasoning/calculation step.

---

## What Still Needs to Be Done

The current implementation was developed and tested with mock OCR data.

When the real OCR/document corpus is available, the next steps are:

1. Connect the Retrieval `/ingest` endpoint to the real Doc Processor output.
2. Verify that the real OCR structure matches the expected input contract.
3. Build the retrieval corpus from the full document collection.
4. Evaluate:

   * Dense retrieval
   * BM25
   * Hybrid RRF
   * Hybrid + Reranking
5. Compare retrieval metrics on the TAT-DQA evaluation set.
6. Tune chunking/retrieval/reranking if needed based on the results.
7. Integrate `/search` with the Agent.

The mock document is only for testing and should not be treated as a production dependency.

---

## Important Integration Flow

```text
PDF
 ↓
Doc Processor / OCR
 ↓
POST /ingest
 ↓
Retrieval
 ↓
User Question
 ↓
Agent
 ↓
POST /search
 ↓
Evidence
 ↓
Agent reasoning
 ↓
Final Answer
```

The Agent mainly needs the `/search` endpoint and the returned evidence schema.

---

## Project Structure

```text
retrieval-api/
├── app/
│   ├── main.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   └── evaluation.py
│
├── experiments/
│   ├── LEDGER_Retrieval_Experiments.ipynb
│   └── results/
│
├── data/
│   └── README.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Current Status

**Implemented:** Retrieval pipeline + FastAPI API + mock end-to-end testing.

**Next:** Real OCR integration, full-corpus evaluation, tuning, and Agent integration.
