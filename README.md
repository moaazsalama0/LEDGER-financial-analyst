# orchestrator-api

This is the piece of LEDGER that everything else talks through. The UI never
calls the agent or the validator directly — it only ever calls this service,
and this service does the rest of the work of getting a question answered
correctly and safely.

## Why this service exists

Three other services do the real thinking:

- **agent-service** figures out how to answer a question — it decides
  whether it needs to look something up, calculate something, or admit it
  doesn't know.
- **answer-validator-api** double-checks that whatever the agent came up
  with is actually well-formed and backed by a real citation, before
  anyone sees it.
- **retrieval-api** (used internally by the agent, not called from here)
  is where the actual document search happens.

None of them know about each other directly. Something has to sit between
the user and all three of them, make the calls in the right order, and
turn three separate HTTP round-trips into one clean answer. That's this
service.

## What actually happens when someone asks a question

1. The UI sends `POST /ask` with just `{"question": "..."}`.
2. This service makes up a `trace_id` — a random ID — so that this one
   question can be followed through every service it touches, in case
   someone needs to debug it later.
3. It calls agent-service's `/query` with the question and that trace_id.
4. Whatever the agent sends back gets forwarded to answer-validator-api's
   `/validate_answer` — not shown to the user yet.
5. The validator either approves it or rejects it.
6. Whatever happened gets turned into one of four possible replies, and
   that's what the UI actually sees.

The four replies are:

- **validated, with an answer** — the normal case. A direct fact, a
  calculated number, or a list of things, plus the document/page it came
  from.
- **validated, but no answer** — this happens when the agent honestly
  couldn't find evidence for the question. That's not a bug. The whole
  point of this project is that the system is supposed to say "I don't
  know" instead of making something up.
- **rejected** — the validator caught something wrong with the shape of
  the answer (missing a citation, wrong type, etc). The reason it was
  rejected gets passed straight through so it's not a mystery.
- **error** — one of the other two services didn't respond at all
  (down, timed out, whatever). This service doesn't crash when that
  happens — it just says clearly which service failed and why.


  ---

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
Orchestrator
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

The UI mainly needs the `/ask` endpoint and the returned response schema described above.


## Running it

You need agent-service and answer-validator-api already running (ports
8004 and 8006) before this will do anything useful.

```
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Once it's up, `http://localhost:8000/docs` gives you a page where you can
type in a question and see the response without needing curl.

`GET /health` is worth checking first if anything seems off — it tells
you plainly whether it can actually reach the other two services or not,
which is usually the real problem when something looks broken.

## Testing it

There are two very different kinds of tests here, and they answer two
different questions.

**"Is the routing logic itself correct?"** — run `pytest`. This fakes out
agent-service and the validator entirely, so it runs instantly and
doesn't need anything else on your machine to be running. It's checking
things like: does a rejected answer actually carry the rejection reason,
does an unreachable agent turn into a clean error instead of a crash,
does every request get its own trace_id.

**"Does it actually work against the real thing?"** — run
`pytest -m integration -v`, but only after starting all three real
services. This sends real questions through the real pipeline and checks
that what comes back is sensible. If the services aren't running, these
tests skip themselves instead of failing, so a plain `pytest` run stays
green no matter what state your machine is in.

## What this isn't finished proving yet

This service itself works — that part is tested and confirmed, including
against the real agent and validator. What it can't fix on its own is
that retrieval-api is still running on sample/mock data rather than the
real, fully-indexed TAT-DQA documents. Until that's connected, most real
questions will correctly come back as "I don't have evidence for that" —
which is the right behavior, just not a very satisfying demo yet. That
part isn't something this service controls.

## Where everything is

```
app/
  main.py               the actual FastAPI app — /ask and /health live here
  orchestrator.py        the step-by-step logic described above
  config.py               all the settings, all overridable via .env
  schemas.py               the shape of every request/response this talks
  logging_config.py        console log formatting
  clients/
    base.py                 shared retry logic used by both service calls
    agent_client.py          talks to agent-service
    validator_client.py      talks to answer-validator-api
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
│   ├── config.py
│   ├── schemas.py
│   ├── logging_config.py
│   └── clients/
│       ├── base.py
│       ├── agent_client.py
│       └── validator_client.py
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
