# Project LEDGER — Evaluation & Observability Service

`eval-service` is the evaluation/observability microservice for Project LEDGER.

This service receives their results, compares benchmark predictions with ground truth and records quality scores in Langfuse.

**Default port:** `8005`

## Responsibilities

- Retrieval evaluation: `Recall@1`, `Recall@3`, `Recall@5`, `MRR`
- Answer evaluation: Exact Match, token F1, scale accuracy, numerical accuracy
- Observability: component latency, validation status, document-processing errors
- Failure analysis: retrieval vs answer vs validator vs document-processing failures
- Langfuse Cloud: component observations and evaluation scores linked by `trace_id`

Ground truth is owned by the evaluation workflow and must never be sent to
retrieval or reasoning before they produce a prediction.

## Folder structure

```text
eval-service/
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py
│   ├── metrics.py
│   └── run_eval.py
├── observability/
│   ├── __init__.py
│   └── langfuse_client.py
├── service/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── store.py
├── tests/
│   ├── __init__.py
│   ├── test_evaluator.py
│   └── test_metrics.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

```bash
cd eval-service
python -m venv .venv
```

Activate the environment, then:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Put your real Langfuse keys in `.env`. `.env` is ignored by Git.

For Langfuse Cloud:

```env
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=development
```

## Run the API

```bash
uvicorn service.main:app --host 0.0.0.0 --port 8005 --reload
```

Swagger/OpenAPI will be available at:

```text
http://localhost:8005/docs
```

Health check:

```bash
curl http://localhost:8005/health
```

## Service event flow

All services must propagate the same project `trace_id` for one request.
`eval-service` accepts arbitrary project trace IDs and deterministically maps them
to the 32-hex W3C trace IDs required by Langfuse.

### 1. Retrieval / reranking event

`POST /events/retrieval`

```json
{
  "trace_id": "trace_123",
  "question_id": "P001",
  "latency": 0.142,
  "evidence": [
    {
      "document_id": "doc_001",
      "chunk_id": "chunk_001",
      "rank": 1,
      "score": 0.92,
      "page_number": 5,
      "content_type": "text"
    }
  ]
}
```

### 2. Agent / reasoning event

`POST /events/agent`

```json
{
  "trace_id": "trace_123",
  "question_id": "P001",
  "latency": 0.8,
  "predicted_answer": "142.5 million",
  "unit": "million",
  "answer_type": "direct",
  "selected_evidence_ids": ["chunk_001"],
  "metadata": {}
}
```

### 3. Validator event

`POST /events/validator`

```json
{
  "trace_id": "trace_123",
  "question_id": "P001",
  "latency": 0.01,
  "status": "validated",
  "reason": null,
  "confidence": 0.98
}
```

### 4. Document processor event

`POST /events/document-processor`

```json
{
  "trace_id": "trace_123",
  "latency": 0.25,
  "document_id": "doc_001",
  "total_page_count": 12,
  "errors": []
}
```

### 5. Evaluate a benchmark trace

Once the pipeline has finished, the benchmark/evaluation runner sends ground truth
only to this endpoint:

`POST /evaluate`

```json
{
  "trace_id": "trace_123",
  "ground_truth": {
    "question_id": "P001",
    "answer": "142.5 million",
    "scale": "million",
    "gold_doc_ids": ["doc_001"]
  }
}
```

Example response:

```json
{
  "trace_id": "trace_123",
  "question_id": "P001",
  "retrieval_metrics": {
    "recall_at_1": 1.0,
    "recall_at_3": 1.0,
    "recall_at_5": 1.0,
    "mrr": 1.0,
    "precision_at_1": 1.0,
    "precision_at_3": 1.0,
    "precision_at_5": 1.0
  },
  "answer_metrics": {
    "exact_match": 1.0,
    "f1": 1.0,
    "scale_accuracy": 1.0,
    "numerical_accuracy": 1.0
  },
  "latency_metrics": {
    "document_processor_latency": 0.25,
    "retrieval_latency": 0.142,
    "agent_latency": 0.8,
    "validator_latency": 0.01,
    "sum_component_latency": 1.202
  },
  "trace_failure_analysis": {
    "retrieval_failure": false,
    "answer_failure": false,
    "validator_rejected": false,
    "document_processing_error": false,
    "missing_components": [],
    "reasons": []
  }
}
```

## Offline batch evaluation

`evaluation/run_eval.py` evaluates **already completed outputs**. It does not own or
run a retriever/reasoner.

Input file format:

```json
[
  {
    "trace_id": "trace_123",
    "question_id": "P001",
    "ground_truth": {
      "answer": "142.5 million",
      "scale": "million",
      "gold_doc_ids": ["doc_001"]
    },
    "retrieval": {
      "latency": 0.142,
      "evidence": [
        {
          "document_id": "doc_001",
          "chunk_id": "chunk_001",
          "rank": 1,
          "score": 0.92
        }
      ]
    },
    "agent": {
      "latency": 0.8,
      "predicted_answer": "142.5 million",
      "unit": "million"
    },
    "validator": {
      "latency": 0.01,
      "status": "validated"
    },
    "document_processor": []
  }
]
```

Run:

```bash
python -m evaluation.run_eval path/to/eval_records.json
```

Run without uploading scores to Langfuse:

```bash
python -m evaluation.run_eval path/to/eval_records.json --no-langfuse
```

Write per-trace details:

```bash
python -m evaluation.run_eval path/to/eval_records.json \
  --details eval_results.json
```

## Tests

```bash
pytest -q
```

## Important implementation notes

1. **Exact Match and numerical accuracy are intentionally separate.**
   `38.4` vs `38.4 million` is not an exact string match, while its numeric value
   can still be correct. Scale accuracy measures the unit separately.
2. Retrieval metrics de-duplicate repeated chunks from the same document before
   calculating document-level Recall@K/MRR.
3. The FastAPI store is intentionally in-memory for team integration/prototyping.
   A process restart clears pending events. Langfuse remains the persistent
   observability backend.
4. No Langfuse key or `.env` file should be committed to Git.
