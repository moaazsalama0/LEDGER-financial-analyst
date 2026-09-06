# answer-validator-api

Project LEDGER — Answer Validator Service (**port 8006**).

The single source of truth for what constitutes a valid, grounded answer.
Validates incoming agent answers against the Strict Answer Schema (`direct`,
`calculated`, `multi_span`, `insufficient_evidence`) before they reach the user.

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app, exposes `POST /validate_answer` |
| `schemas.py` | Pydantic models for the request envelope and responses |
| `validators.py` | Per-type validation rules and rejection reasons |
| `logger_config.py` | Formats the `[ANSWER-VALIDATOR-SUCCESS/ERROR]` console logs |
| `tests/test_validator.py` | Valid + invalid test case per answer type |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8006 --reload
```

Swagger UI: http://localhost:8006/docs
Health check: `GET /health`

## Run tests

```bash
pytest tests/ -v
```

## Endpoint

### `POST /validate_answer`

**Request body** — one of the four answer types:

```json
{
  "answer_type": "direct",
  "evidence": [{ "document_id": "doc_017", "page": 1, "section": "Income Statement" }],
  "params": { "value": "$142.5M" }
}
```

**Success response** (`200`):
```json
{ "status": "validated", "answer": { ...same payload... } }
```

**Rejected response** (`422`):
```json
{ "status": "rejected", "reason": "Missing required evidence citation." }
```

## Validation rules by type

| Type | Required `params` | Evidence |
|---|---|---|
| `direct` | `value` (string or number) | ≥1 citation required |
| `calculated` | `value` (number), `formula` (string) | ≥1 citation required |
| `multi_span` | `values` (array, 2+ items) | ≥1 citation required |
| `insufficient_evidence` | `reason` (string) | optional, may be empty |

Every evidence item must have a non-empty string `document_id` and an integer `page`.

## Console log formats

Success:
```
[ANSWER-VALIDATOR-SUCCESS] Received and validated answer of type 'calculated' with evidence {'document_id': 'doc_041', 'page': 2}
```

Error (envelope-level, e.g. missing evidence or bad `answer_type`):
```
[ANSWER-VALIDATOR-ERROR] Invalid answer. Reason: Missing required evidence citation.
```

Error (type-specific, e.g. missing a required param):
```
[ANSWER-VALIDATOR-ERROR] Invalid answer for 'calculated': Missing required key 'formula'.
```

## Known simplification (documented per spec's "any addition must be documented" rule)

The spec states `calculated` answers require "one citation per operand used in
the formula." Operands actually used from evidence vs. constants in the
formula (e.g. the `*100` in a percent-change calc) can't be reliably told
apart from the formula string alone, so this service only enforces "at least
one evidence citation" for `calculated` answers rather than an exact operand
count match. If the agent-service starts tracking operand-to-evidence
mapping explicitly, tighten `validators.py` accordingly.
