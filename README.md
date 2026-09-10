# orchestrator-api

This is the piece of LEDGER that everything else talks through. The UI never
calls the agent, the validator, the document processor, or retrieval
directly — it only ever calls this service, and this service does the rest
of the work of getting a question answered correctly and safely, or a
document ingested correctly.

## Why this service exists

Four other services do the real thinking:

- **agent-service** figures out how to answer a question — it decides
  whether it needs to look something up, calculate something, or admit it
  doesn't know.
- **answer-validator-api** double-checks that whatever the agent came up
  with is actually well-formed and backed by a real citation, before
  anyone sees it.
- **doc-processor-api** turns a raw uploaded PDF into structured text,
  headings, and tables.
- **retrieval-api** indexes that structured content (used internally by
  the agent for search, and also used by this service to ingest new
  documents and to list what's indexed for the dashboard).

None of them know about each other directly. Something has to sit between
the user and all of them, make the calls in the right order, and turn
several separate HTTP round-trips into one clean answer. That's this
service.

## What actually happens when someone asks a question

1. The UI sends `POST /ask` with just `{"question": "..."}`.
2. This service makes up a `trace_id` — a random ID — so that this one
   question can be followed through every service it touches (including
   later by eval-service), in case someone needs to debug it later.
3. It calls agent-service's `/query` with the question and that trace_id.
4. Whatever the agent sends back gets forwarded to answer-validator-api's
   `/validate_answer` — not shown to the user yet. The round-trip latency
   of this validator call is measured and reported to eval-service.
5. The validator either approves it or rejects it.
6. Whatever happened gets turned into one of four possible replies, and
   that's what the UI actually sees.

The four replies are:

- **validated, with an answer** — the normal case. A direct fact, a
  calculated number, or a list of things, plus the document/page it came
  from. The reply also carries `answer_type` and `params` (the full
  typed payload the agent produced), for a UI that wants to render
  calculated/multi_span answers differently from a plain fact.
- **validated, but no answer** — this happens when the agent honestly
  couldn't find evidence for the question. That's not a bug. The whole
  point of this project is that the system is supposed to say "I don't
  know" instead of making something up.
- **rejected** — the validator caught something wrong with the shape of
  the answer (missing a citation, wrong type, etc). The reason it was
  rejected gets passed straight through so it's not a mystery.
- **error** — one of the other services didn't respond at all
  (down, timed out, whatever). This service doesn't crash when that
  happens — it just says clearly which service failed and why.

Every evidence object carries both `page` and `page_number` (some services
in the pipeline use one spelling, some the other), plus `content_type` and
`snippet` when the agent provides them.

## What happens when someone uploads a document

1. The UI sends `POST /documents/upload` with a raw PDF file.
2. This service forwards the file to doc-processor-api's `POST /process`,
   which returns structured text, headings, and tables.
3. This service adapts that structured output into the shape
   retrieval-api's `POST /ingest` expects (the two contracts don't line up
   field-for-field, so this is the one place that reconciles the
   difference — the raw processor output is never forwarded as-is).
4. It sends the adapted payload to retrieval-api's `POST /ingest`.
5. On success, the document is recorded in an in-memory registry (title,
   page count, detected tables with a few sample cell values) so the
   dashboard can show it, and a simple `{"status": "success", "document_id": ...}`
   goes back to the UI. Any failure at either step returns
   `{"status": "error", "stage": "processor"|"retrieval", "message": ...}`
   instead — this flow never touches `/ask`'s logic.

## Dashboard support: `GET /documents`

Added for ui-service's Documents/Dashboard tabs (not part of the official
Contracts PDF). Returns `{"count": N, "documents": [...]}`, where each
document has `document_id`, `document_title`, `filename`, `page_count`,
`table_count`, a few sample table values, and `indexed_at`.

This merges two sources:
- the in-memory registry above (documents uploaded through **this**
  orchestrator process's own `/documents/upload` — resets on restart,
  but has rich metadata)
- retrieval-api's own document listing, if it has one, so documents
  indexed outside orchestrator (e.g. a batch-ingested corpus) show up too

**Status of that second source: unconfirmed.** `RETRIEVAL_DOCUMENTS_ENDPOINT`
(`/documents` on retrieval-api, by assumption) is not in the official
Contracts PDF. If it doesn't exist yet, isn't reachable, or returns an
unrecognized shape, this falls back to the local registry alone rather
than erroring — it's a best-effort dashboard enrichment, not a step in
the critical `/ask` or upload path. **Needs confirmation from the
retrieval-api owner**: does this endpoint exist, and if so at what path
and in what shape.

## Important Integration Flow

```text
User
 ↓
UI
 ↓
POST /ask
 ↓
Orchestrator
 ↓
POST /query → Agent-service
 ↓
Agent-service (internally calls Retrieval + Calculator as needed)
 ↓
Answer
 ↓
Orchestrator  ── emits validator-latency event → eval-service
 ↓
POST /validate_answer → Answer-validator-api
 ↓
Validated / Rejected
 ↓
Orchestrator
 ↓
Final response
 ↓
UI
```

```text
User
 ↓
UI  (upload PDF)
 ↓
POST /documents/upload
 ↓
Orchestrator
 ↓
POST /process → doc-processor-api
 ↓
Orchestrator (adapts processor output → retrieval-api's ingest shape)
 ↓
POST /ingest → retrieval-api
 ↓
Orchestrator (records document in local registry)
 ↓
Success / Error response
 ↓
UI
```

The UI mainly needs `/ask`, `/documents/upload`, `/documents`, and the
response schemas described above.

## Running it

You need agent-service and answer-validator-api already running (ports
8004 and 8006) for `/ask` to do anything useful. For document upload,
doc-processor-api (8008) and retrieval-api (8001) need to be running too.
eval-service (8005) is optional — if it's not running, the telemetry
calls fail silently (logged at DEBUG only) and nothing else is affected.

```
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Once it's up, `http://localhost:8000/docs` gives you a page where you can
try every endpoint (`/ask`, `/documents/upload`, `/documents`) without
needing curl.

`GET /health` is worth checking first if anything seems off — it tells
you plainly whether it can actually reach agent-service, answer-validator-api,
retrieval-api, and doc-processor-api (checked via `/ready`, since a process
that's up but still loading its OCR model needs to show as "degraded", not
"reachable").

## Testing it

There are two very different kinds of tests here, and they answer two
different questions.

**"Is the routing logic itself correct?"** — run `pytest`. This fakes out
every downstream service entirely, so it runs instantly and doesn't need
anything else on your machine to be running.

**"Does it actually work against the real thing?"** — run
`pytest -m integration -v`, but only after starting the real services.
If they aren't running, these tests skip themselves instead of failing.

## What this isn't finished proving yet

This service itself works — routing, validation, upload, and the
dashboard endpoint are all tested and confirmed against real
answer-validator-api. Two things outside this service's control are still
open:

- **agent-service**: currently blocked for live end-to-end testing —
  its configured Gemini model (`gemini-2.0-flash`) is no longer available.
  Needs the agent-service owner to update the model config.
- **retrieval-api's document listing**: as described above, `GET /documents`
  works today off the local registry alone; showing the *full* corpus on
  the dashboard needs confirmation (or a small addition) from
  retrieval-api's owner.

## Where everything is

```
app/
  main.py               /health, /ask, /documents/upload, /documents
  orchestrator.py        the step-by-step logic described above
  telemetry.py            fire-and-forget validator-latency events → eval-service
  config.py               all the settings, all overridable via .env
  schemas.py               the shape of every request/response this talks
  logging_config.py        console log formatting
  clients/
    base.py                 shared retry logic used by service calls
    agent_client.py          talks to agent-service
    validator_client.py      talks to answer-validator-api
    processor_client.py      talks to doc-processor-api
    retrieval_client.py      talks to retrieval-api (ingest + document listing)
tests/                    both kinds of tests described above
requirements.txt          what you need to run it
requirements-dev.txt      what you additionally need to test it
.env.example              copy this to .env and edit if your ports differ
```

---

## File Structure

```text
orchestrator-api/
├── app/
│   ├── main.py
│   ├── orchestrator.py
│   ├── telemetry.py
│   ├── config.py
│   ├── schemas.py
│   ├── logging_config.py
│   └── clients/
│       ├── base.py
│       ├── agent_client.py
│       ├── validator_client.py
│       ├── processor_client.py
│       └── retrieval_client.py
│
├── tests/
│   ├── conftest.py
│   ├── test_schemas.py
│   ├── test_orchestrator_unit.py
│   ├── test_api.py
│   └── test_integration_live.py
│
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md
```