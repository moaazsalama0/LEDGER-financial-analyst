# ui-service

**The face of LEDGER — the only part of the system a person actually looks at.**


---

## Table of Contents

- [What this service is](#what-this-service-is)
- [Why it exists](#why-it-exists)
- [The one rule this service follows](#the-one-rule-this-service-follows)
- [What's on screen](#whats-on-screen)
- [How a question actually flows](#how-a-question-actually-flows)
- [The orchestrator contract, as seen from here](#the-orchestrator-contract-as-seen-from-here)
- [The bonus feature: evidence highlighting](#the-bonus-feature-evidence-highlighting)
- [Design decisions worth knowing about](#design-decisions-worth-knowing-about)
- [Where everything lives](#where-everything-lives)
- [Configuration](#configuration)
- [Getting it running](#getting-it-running)
- [Known limitations](#known-limitations)
- [Quick troubleshooting](#quick-troubleshooting)

---

## What this service is

`ui-service` is the Gradio app a person actually sits in front of. Every
other LEDGER service — agent-service, retrieval-api, doc-processor-api,
answer-validator-api — does its work invisibly, behind `orchestrator-api`.
This is the one place all of that becomes a chat box, an upload button,
and a dashboard someone can actually read.

It is deliberately thin. It has no opinion about how a question should be
answered, what counts as evidence, or how a PDF should be parsed — it just
asks `orchestrator-api` and displays whatever comes back, honestly.

## Why it exists

Without this service, "using LEDGER" would mean sending raw HTTP requests
to `orchestrator-api` and reading JSON. This service exists to turn that
into something a non-technical person could sit down with: type a
question, get an answer with a citation, see the actual cited page
highlighted, upload a new report, and check at a glance whether the whole
pipeline is healthy.

## The one rule this service follows

**ui-service talks to exactly one other service: `orchestrator-api`.**

It never calls agent-service, retrieval-api, doc-processor-api, or
answer-validator-api directly, and it isn't supposed to know they exist.
`app/orchestrator_client.py` is the *only* file in this codebase allowed
to know an HTTP endpoint path — every other module gets back plain
Python objects with the network plumbing already resolved. If this
service ever needs to point at a different host or port for the
orchestrator, `app/config.py` is the one place that changes.

## What's on screen

| Tab | What it does |
|---|---|
| **ℹ️ About LEDGER** | A plain-language explanation of what the project is and what each tab does — the on-ramp for a first-time visitor. |
| **💬 Ask LEDGER** | The chat interface. Ask a question across the whole indexed corpus, or scope it to one document. Every answer shows its citation, and the **Evidence Inspector** panel renders the actual cited PDF page alongside it. |
| **📁 Documents** | Upload raw PDFs. Each one is routed through `orchestrator-api → doc-processor-api → retrieval-api` for real OCR/layout extraction and indexing — never a shortcut through any pre-parsed dataset JSON. |
| **📊 Dashboard** | Documents indexed, questions asked, validated-answer rate, average latency, recent query log, detected tables, and extracted values — see [Known limitations](#known-limitations) for exactly what this can and can't see. |
| **🩺 System Health** | Pings `orchestrator-api`, which in turn reports whether agent-service and answer-validator-api are reachable. |

## How a question actually flows

1. Someone types a question into the **Ask LEDGER** tab (optionally
   scoped to one document).
2. `ui.py` hands it to `OrchestratorClient.ask()`.
3. That sends `POST /ask` to `orchestrator-api` and waits.
4. Whatever comes back — validated, rejected, or an error — gets turned
   into plain text by `formatting.py` and dropped into the chat.
5. If the answer carries evidence, `evidence_renderer.py` tries to render
   the actual cited PDF page (if that document was uploaded during this
   session) with the evidence highlighted in yellow.
6. The exchange is logged to an in-memory `QueryLogStore` so the
   **Dashboard** tab has something to show.

Uploading a document follows the same shape, just through
`POST /documents/upload` and a `DocumentStore` instead.

## The orchestrator contract, as seen from here

This is everything `ui-service` knows about `orchestrator-api` — the
entire contract lives in `orchestrator_client.py`, nowhere else:

```
GET  /health
  -> {"status": "ok", "dependencies": {"agent-service": "reachable" | "degraded" | "unreachable", ...}}

POST /ask                body: {"question": "<str>"}
  -> {"status": "validated" | "rejected" | "error",
      "answer": <str|number|list|None>,
      "evidence": [{"document_id", "page"/"page_number", "section"}, ...],
      "reason": <str|None>,
      "trace_id": <str|None>}

POST /documents/upload   multipart file
  -> {"status": "success", "document_id": "...", "message": "..."}
  or {"status": "error", "stage": "processor" | "retrieval", "message": "..."}
```

Every call is wrapped so a bad response, a timeout, or a dead
orchestrator never crashes the UI — it always comes back as a normal
`AskResult` / `UploadResult` / `HealthResult` object with `ok=False` and
a human-readable reason, which the UI just displays like any other
answer.

## The bonus feature: evidence highlighting

The Strict Answer Schema's evidence objects only carry
`{document_id, page, section}` — no bounding box travels through
`agent-service → answer-validator-api → orchestrator-api`. Rather than
skip the "show me where on the page" experience for that reason,
`evidence_renderer.py` does the following:

1. Takes the actual cited PDF page (cached from whatever the user
   uploaded earlier in this session).
2. Uses PyMuPDF's own text search on that page to find where the
   answer's value actually appears.
3. Highlights it in soft yellow and turns the page into an image.
4. If the exact answer text isn't found, it falls back to highlighting
   the `section` heading instead — so there's still a "look here."
5. If the cited document was never uploaded in this session, it says so
   plainly instead of faking an image.

## Design decisions worth knowing about

- **One downstream dependency, no exceptions.** Every other service is
  invisible to this one on purpose — it keeps the UI simple to reason
  about and impossible to accidentally couple to another service's
  internal API.
- **The Dashboard is an honest mirror of this session, not the corpus.**
  `orchestrator-api`'s contract only exposes `/ask` and
  `/documents/upload` — there is no "list everything ever indexed"
  endpoint anywhere in the pipeline. So the Dashboard can only show what
  has actually passed through *this* `ui-service` session. That's
  documented here rather than quietly faked.
- **Every network call resolves to a plain object, never an exception.**
  `orchestrator_client.py` catches timeouts and connection errors itself
  and returns a normal result object with `ok=False` — Gradio callbacks
  never need a `try/except` of their own.
- **In-memory, single-process state is a deliberate, scoped choice.**
  `DocumentStore` and `QueryLogStore` are plain Python lists, which is
  enough for a single-process Gradio demo — not a design meant to
  survive a restart or scale past one session.

## Where everything lives

```text
ui-service/
├── app/
│   ├── main.py                 entry point — python -m app.main
│   ├── config.py                all settings, overridable via .env
│   ├── ui.py                     the Gradio Blocks layout + all callbacks
│   ├── orchestrator_client.py    the ONLY file that knows an HTTP endpoint path
│   ├── formatting.py             turns AskResult/UploadResult into chat Markdown
│   ├── evidence_renderer.py      the bounding-box PDF highlight bonus feature
│   └── stores.py                 in-memory DocumentStore + QueryLogStore
│
├── requirements.txt             gradio, httpx, pydantic-settings, pandas, pymupdf, pillow
├── run.sh                        convenience launcher (creates venv, installs, runs)
├── .env.example                  copy this to .env to override any default
└── README.md                    you're reading it
```

## Configuration

Everything below lives in `app/config.py`, overridable via `.env`
(copy `.env.example` to start):

| Variable | Purpose | Default |
|---|---|---|
| `HOST` | ui-service's own bind address | `0.0.0.0` |
| `PORT` | ui-service's own port | `8007` |
| `ORCHESTRATOR_URL` | Base URL for orchestrator-api — the only downstream dependency | `http://localhost:8000` |
| `REQUEST_TIMEOUT_SECONDS` | Timeout for `/ask` and upload calls | `60` |
| `HEALTH_TIMEOUT_SECONDS` | Timeout for `/health` checks | `5` |
| `EVIDENCE_RENDER_DPI` | Resolution used when rendering a cited PDF page for the highlight preview | `150` |
| `APP_TITLE` | Title shown in the browser tab / header | `LEDGER — Financial Document Intelligence Agent` |
| `SHARE` | Whether Gradio generates a public share link | `False` |

## Getting it running

`orchestrator-api` must already be running (default `http://localhost:8000`)
before this is useful — without it, every question and upload will come
back as a connection error.

```bash
cd ui-service
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.main
```



Once it's up: **http://localhost:8007**

## Known limitations

- **`list_documents()` does not exist yet.** The current `ui.py` calls
  `self.client.list_documents()` in three places (the merged documents
  table, the detected-tables table, and the document-scope dropdown),
  but `orchestrator_client.py` only implements `check_health()`, `ask()`,
  and `upload_document()`. As written today, opening the **Dashboard**
  tab or asking a question will raise an `AttributeError`. This needs a
  `list_documents()` method added to `OrchestratorClient` (and, most
  likely, a `GET /documents` endpoint added to `orchestrator-api` itself,
  since that endpoint isn't in its current contract) before this version
  of `ui.py` can run end-to-end.
- **The Dashboard only sees this session.** By design — see
  [Design decisions](#design-decisions-worth-knowing-about) — it is not
  a view of the whole indexed corpus unless `list_documents()` above is
  wired up to a real endpoint.
- **No automated tests yet.** There's no `tests/` directory in this
  service at the moment.

## Quick troubleshooting

| Symptom | Likely cause |
|---|---|
| Every question/upload fails immediately | `orchestrator-api` isn't running, or `ORCHESTRATOR_URL` in `.env` points at the wrong host/port. Check the **System Health** tab first. |
| App crashes opening the Dashboard or right after asking a question | The missing `list_documents()` method above — expected until it's implemented. |
| Evidence Inspector says the PDF isn't cached | The cited document wasn't uploaded through this same session — upload it in the **Documents** tab first, then re-ask. |
| Port 8007 already in use | Another `ui-service` instance is still running — stop it, or change `PORT` in `.env`. |
