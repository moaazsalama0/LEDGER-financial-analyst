# doc-processor-api

LEDGER service 2. Takes a raw financial-report PDF and returns a structured
representation: text, tables, headings, page numbers, and bounding boxes.

It knows nothing about retrieval, reasoning, or answers. Its only job is to
turn a PDF into something the retrieval service can index without ever
reopening the PDF.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt         # Linux/macOS

.venv/Scripts/python.exe -m pytest -q                          # 438 tests, ~7s, CPU only
```

The service runs on Docling, which is the production document-processing path
and the only backend it serves. Install the weights, then start it:

```bash
.venv/Scripts/python.exe -m pip install -r requirements-docling.txt
cp .env.example .env                                           # optional
.venv/Scripts/python.exe -m app.main                           # serves DOC_PORT (8008)
```

Or drive uvicorn yourself, which is what a container or a process manager
usually does — `--port` wins, because that path never reads `DOC_PORT`:

```bash
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8008
```

Every setting has a working default, so the `.env` step is optional —
`.env.example` is there so the whole configuration surface is visible in one
place, with the reason each knob exists. It is read relative to the working
directory, and real environment variables override it.

The weights download on first use, so give the first request a few minutes —
or call `GET /ready`, which reports `false` until they are loaded.

One trap worth knowing before you plan a corpus run: on Windows, plain
`pip install docling` resolves torch to the **CPU-only** wheel. It works, and
it is slow enough to change the plan — TableFormer in accurate mode is minutes
per document on CPU. Check with
`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`,
and if it says `+cpu` on a machine with an NVIDIA card, see the instructions in
`requirements-docling.txt`. No code changes either way: docling's accelerator
device is `auto`.

Then:

```bash
curl -F file=@report.pdf -F document_id=doc_0417 \
     http://127.0.0.1:8008/process | jq
```

Note: on a machine with MSYS2 on `PATH`, `python` may resolve to the mingw
build, whose wheel tag is `mingw_x86_64_msvcrt_gnu`. No PyPI wheels exist for
it, so `pydantic-core` tries to compile from source and fails. Build the venv
with a native CPython (`win-amd64`) instead.

---

## For the retrieval team

Install the contract package and code against the types, not against raw JSON:

```bash
pip install ./contracts
```

```python
from ledger_doc_contract.v1 import ProcessedDocument

doc = ProcessedDocument.model_validate(response.json())

for block in doc.iter_blocks():          # reading order, furniture excluded
    chunk_metadata = {
        "document_id": doc.document_id,
        "page":        block.page_number,
        "section":     doc.section_path(block.section_id),   # "Notes > 12. Income Taxes"
        "content_type": block.type.value,
    }

for table in doc.tables:                 # for search_tables()
    embed(table.markdown, metadata={"table_id": table.table_id, ...})
```

The four metadata fields the brief requires on every chunk are all present on
every block. `" > ".join(section.path)` — which `doc.section_path()` does for
you — is exactly the `section` string the Strict Answer Schema wants in an
evidence citation.

### Guarantees

1. **Stable ids.** `block_id`, `section_id`, `table_id`, and `cell_id` are
   deterministic for a given `(content_sha256, backend, service minor
   version)`. Safe as vector-store primary keys.
2. **Uniform geometry.** Every `bbox` is in PDF points, top-left origin, y
   increasing downward, and each page carries its `width`/`height`. Usable
   directly for bounding-box highlighting.
3. **Determinism.** The same PDF yields a byte-identical document body.
   `processing` is the only part that varies between runs — exclude it when
   hashing, or use `doc.to_stable_dict()`.
4. **Additive-only within 1.x.** Fields may be added; existing fields never
   change type or disappear. A breaking change bumps to `v2` and both are
   served during migration.
5. **Two access paths.** Walk `pages[].blocks[]` for linear chunking, or
   iterate `tables[]` for table search. Tables appear in both, joined by
   `table_id`.
6. **Nothing is destroyed.** A cell's printed text is always preserved beside
   its parsed value, so you can redo any parse you disagree with.

### Where the boundary sits

Chunking belongs to `retrieval-api`, not here. Chunk size and overlap are
retrieval-tuning parameters and must be changeable without re-parsing 2,758
PDFs. Our job is to make chunking possible without a PDF parser on your side:
section-coherent blocks, whole tables as atomic units, and a section tree.

---

## API

### `POST /process`

`multipart/form-data`

| Field | Required | Notes |
|---|---|---|
| `file` | yes | PDF. Verified by magic bytes, not by the declared content type. Max 50 MB, 100 pages. |
| `document_id` | no | `^[A-Za-z0-9_.-]{1,128}$`. Defaults to `doc_<sha256[:16]>`, so re-ingesting the same PDF is idempotent. |
| `options` | no | JSON: `{"backend": "docling", "force_ocr": false}` |

Only two options, each earning its place: `force_ocr` because some pages are
scanned and the classifier can misjudge them, and `backend` because the brief
requires comparing pipeline variants with measured results. Unknown options
are rejected rather than ignored, so a misspelled flag fails immediately
instead of silently doing nothing.

### Errors

One body shape for every failure:

```json
{"error_code": "INVALID_PDF", "message": "...", "document_id": null, "detail": {}}
```

Branch on `error_code`.

| Status | `error_code` | Cause |
|---|---|---|
| 400 | `INVALID_PDF` | Not a PDF, corrupt, or zero pages |
| 400 | `ENCRYPTED_PDF` | Password-protected |
| 400 | `INVALID_DOCUMENT_ID` | Fails the id pattern |
| 413 | `FILE_TOO_LARGE` | Over `DOC_MAX_UPLOAD_MB` |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | Not declared as a PDF |
| 422 | `TOO_MANY_PAGES` | Over `DOC_MAX_PAGES` |
| 422 | `INVALID_OPTIONS` | Malformed or unknown option |
| 500 | `BACKEND_FAILURE` | The backend failed on the whole run |
| 503 | `BACKEND_UNAVAILABLE` | Unknown backend, or models not loaded |

A single unreadable page never fails the document: it is returned with
`extraction_source: "failed"` and a `processing.warnings` entry.

### Other endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Process is up |
| `GET /ready` | Backend models are loaded |
| `GET /version` | Service, backend, model, and contract versions |
| `GET /contract/schema` | JSON Schema, generated from the live models |

---

## Output shape

```jsonc
{
  "contract_version": "1.0",
  "document_id": "doc_0417",
  "content_sha256": "9f2c…",
  "page_count": 3,
  "sections": [
    { "section_id": "s002", "title": "12. Income Taxes", "level": 2,
      "parent_section_id": "s000",
      "path": ["Annual Report 2019", "12. Income Taxes"],
      "page_start": 2, "page_end": 3 }
  ],
  "pages": [
    { "page_number": 2, "width": 612.0, "height": 792.0,
      "extraction_source": "text_layer",
      "ocr_confidence": null,          // null = not measured, not "perfect"
      "blocks": [
        { "block_id": "p0002_b000", "type": "section_header",
          "text": "12. Income Taxes", "heading_level": 2,
          "section_id": "s002", "order_index": 0,
          "bbox": {"x0": 72.0, "y0": 61.9, "x1": 182.7, "y1": 72.2} }
      ],
      "reading_order": ["p0002_b000", "p0002_b001"]   // furniture excluded
    }
  ],
  "tables": [
    { "table_id": "t000", "block_id": "p0002_b002", "page_number": 2,
      "n_rows": 3, "n_cols": 3, "header_rows": 1, "header_cols": 1,
      "column_headers": ["", "2019", "2018"],
      "row_headers": ["Current federal", "Deferred"],
      "units": {"scale": 1000000.0, "scale_label": "in millions", "currency": "USD"},
      "markdown": "|  | 2019 | 2018 |\n|---|---|---|\n…",
      "cells": [
        { "cell_id": "t000_r1_c2", "row": 1, "col": 2,
          "text": "(567)",                  // always as printed
          "value": { "num": -567.0, "scaled": -567000000.0,
                     "is_negative": true, "negative_style": "parentheses" } }
      ]
    }
  ],
  "processing": { "backend": "docling", "duration_ms": 7681, "warnings": [] }
}
```

### Why the numeric parsing matters

Nothing else in the pipeline reads a financial figure the way a filing writes
one. Handled here, once:

| Printed | `num` | Why |
|---|---|---|
| `(1,234)` | −1234.0 | Parentheses are the finance negative. Miss it and the sign is wrong. |
| `$1,234` | 1234.0 | `currency: "USD"` |
| `12.3%` | 12.3 | `unit: "percent"`, and never scaled by "in millions" |
| `—` / `N/A` | `null` | `is_dash: true`. Nil, **not** zero — summing dashes as zero silently changes a total. |
| `1,234 (1)` | 1234.0 | `footnote_refs: ["1"]` |
| caption "in millions" | — | `scaled` on every cell |

---

## Architecture

```
upload → validate (type, size, magic bytes, page count, encryption)
       → probe: text layer present, or scanned?
       → LayoutBackend.parse()          ← the swappable seam
       → measure type from the text layer inside each block the model found
       → normalise geometry → drop page furniture → reading order (LTR, column-aware)
       → heading levels → section tree (spans pages)
       → table grid → caption/units → numeric cell parsing
       → assemble contract → cache → respond
```

A backend returns `RawParse` — a model-agnostic description of what is on each
page — and nothing downstream knows which model produced it. That is what
makes the backend replaceable without touching the API layer, and what lets
the whole test suite run against a no-model reader with no GPU.

### Backends

| Name | Status | Notes |
|---|---|---|
| `docling` | **the production path, and the only one served** | DocLayNet layout model + TableFormer cell structure. This is the backend the deep-learning requirement is about. Needs `requirements-docling.txt`. |
| `textonly` | **retired after Phase 2** | The no-model text-layer control the Phase 2 comparison was measured against. Moved to `evaluation/textonly_baseline.py`; the service does not register it, and `{"backend": "textonly"}` is now refused with `BACKEND_UNAVAILABLE`. Kept so the frozen Phase 2 arm stays reproducible, and because it drives the fast test suite. |
| `surya` | possible | The seam is unchanged, so a second backend is a registration rather than a rewrite. Nothing is registered speculatively: an entry in the registry is a claim the service will serve it. |

`GET /version` reports exactly one:

```json
{ "service": "doc-processor-api", "contract_version": "1.0",
  "default_backend": "docling", "available_backends": ["docling"] }
```

#### How the Docling backend reads a page

Two models do the work no heuristic can: a DocLayNet-trained layout model
classifies each region (title, section header, text, list item, table,
caption, footnote, page furniture), and TableFormer recovers table
*structure* — the cell grid, row and column spans, and which cells are
headers.

Glyphs come from the PDF's own text layer on born-digital pages and from OCR
on scanned ones. That is deliberate, not a shortcut: force-OCR'ing a page
whose text layer is already character-exact can only introduce errors, and
they would land in exactly the figures the answer validator checks. Set
`options.force_ocr` to override it.

One thing the layout model does not report is **type size**, and heading
*levels* depend on nothing else — a document's hierarchy is expressed
typographically or not at all. So `app/processing/typography.py` measures the
type from the text-layer glyphs falling inside each box the model drew. The
model decides what a region is; the glyphs say how it looks. On a scanned page
there are no glyphs, and the probe reports "not measured" rather than a
fabricated default, leaving level assignment to fall back to numbering depth.

`options.backend` may still name `docling` per request, and `DOC_BACKEND`
still selects it, but there is nothing else to select. A name the service does
not serve fails — at startup if it is configured, with `BACKEND_UNAVAILABLE`
if it is requested — rather than silently falling back: a service quietly
running a different backend than asked for produces results nobody can
reproduce.

### Configuration

All `DOC_`-prefixed:

| Variable | Default | Meaning |
|---|---|---|
| `DOC_PORT` | `8008` | Port `python -m app.main` serves on. `uvicorn --port` wins when uvicorn is invoked directly. The host is not settable — it stays a uvicorn argument. |
| `DOC_BACKEND` | `docling` | Layout backend. `docling` is the only supported value; anything else is refused rather than substituted. |
| `DOC_MAX_UPLOAD_MB` | `50` | Upload size limit |
| `DOC_MAX_PAGES` | `100` | Page limit per request |
| `DOC_TEXT_DENSITY_THRESHOLD` | `30` | Chars/page below which a page counts as scanned |
| `DOC_LOW_CONFIDENCE_THRESHOLD` | `0.75` | OCR confidence below which a page is flagged |
| `DOC_RENDER_DPI` | `300` | Rasterisation DPI for OCR paths |
| `DOC_CACHE_ENABLED` | `true` | Content-addressed result cache |
| `DOC_CACHE_DIR` | `.cache/documents` | Cache location |
| `DOC_DOCLING_TABLE_MODE` | `accurate` | TableFormer mode, `accurate` or `fast` |
| `DOC_DOCLING_OCR` | `true` | Whether the Docling pipeline may OCR |
| `DOC_DOCLING_ARTIFACTS_PATH` | unset | Local model directory, for an air-gapped deployment |
| `DOC_DOCLING_THREADS` | `4` | Inference thread count |

`DOC_DOCLING_TABLE_MODE` is recorded in every document's
`processing.model_versions`, because `accurate` and `fast` recover different
grids and score differently — a backend comparison is only honest if each
result says what actually ran.

### Caching

Keyed on `sha256(pdf bytes)` + a *variant* covering the backend's name and the
models it is configured with. The models matter as much as the name:
TableFormer's `accurate` and `fast` modes are different systems that recover
different grids, and serving one's parse for a request that asked for the
other is exactly the unreproducible result the rest of this service goes out of
its way to prevent. The variant is derived from the backend's own
`model_versions`, so any reconfiguration invalidates the cache automatically
and the cache module never has to be taught what a backend is.

For a 2,758-document corpus on a GPU shared with the agent's language model,
the intent is to parse the corpus once offline and ship the cache, so demo-time
ingestion is instant and the GPU stays free. A `force_ocr` run is a different
parse and is neither served from nor written to the same key.

---

## Testing

```bash
.venv/Scripts/python.exe -m pytest -q
```

| Suite | Covers |
|---|---|
| `tests/unit` | Numeric parsing, heading levels, section tree, reading order, table grids and spans, contract assembly, the Docling mapping, the typography probe, the evaluation harness's matching rules |
| `tests/integration` | The HTTP surface and the full error matrix |
| `tests/contract` | Schema stability, round-tripping, determinism |
| `tests/heavy` | The Docling backend against real weights — **excluded by default** |

Everything in the default run works on CPU with no model download and no
network, in a few seconds. The HTTP tests drive the service with the retired
Phase 2 text-layer reader (`evaluation/textonly_baseline.py`), registered by
the test harness and never by the service — that is what keeps the fast suite
model-free while still parsing a real PDF end to end. What the service itself
offers is asserted separately, against the registry exactly as it ships.

The Docling *mapping* — label translation, the coordinate flip, span
arithmetic, header detection — is tested in the fast suite against duck-typed
fakes, with no docling import anywhere in the module's top level. That is the
half most likely to be wrong and it costs milliseconds to check. A missed
coordinate flip puts every citation highlight on the mirror image of the right
paragraph, and no test of the model itself would catch it.

The half that genuinely needs weights runs separately:

```bash
.venv/Scripts/python.exe -m pytest tests/heavy -m heavy -q

# add the table assertions a synthetic fixture cannot support —
# a hand-built PDF has no ruled lines for TableFormer to find
DOC_TEST_PDF=/path/to/report.pdf .venv/Scripts/python.exe -m pytest tests/heavy -m heavy -q
```

After any deliberate contract change:

```bash
.venv/Scripts/python.exe scripts/generate_schema.py
```

The committed `schema.json` is the record of what consumers rely on. The
stability test fails on a removed or retyped field, so regenerating is how a
change gets consciously accepted instead of happening silently.

---

## Looking at what the parser actually found

```bash
.venv/Scripts/python.exe scripts/inspect_document.py report.pdf --backend docling

# the retired Phase 2 control, for comparing a historical parse. This script
# registers it; the service does not.
.venv/Scripts/python.exe scripts/inspect_document.py report.pdf --backend textonly
```

Prints the block-type histogram, every table with its shape and scale, the
section tree, per-page extraction source, and any warnings — plus
`parsed nums`, the count of table cells whose figure was both recovered and
parsed. That last number is the one that predicts arithmetic-question success.

This is the tool for looking at *one* document. Scoring a corpus against the
gold answers is the extraction harness below.

---

## Extraction quality, measured

A demo proves a pipeline can work once. The question a backend choice turns on
is how often it works, and that needs a number produced the same way every
time.

```bash
python scripts/download_tatdqa.py --splits dev          # 274 docs, 1,644 questions, 156 MB

python scripts/run_extraction_eval.py --backend docling
python scripts/run_extraction_eval.py --backend textonly   # the Phase 2 control

python scripts/run_extraction_eval.py --compare \
    runs/dev-textonly.jsonl runs/dev-docling.jsonl
```

### Result: the backend choice, settled

Full TAT-DQA dev split — 274 documents, 307 pages, 1,644 questions, 2,804
gold facts. Both backends scored the identical documents by the identical
rule.

|  | `textonly` | `docling` |
|---|---|---|
| **question support** | 19.7% | **85.3%** |
| fact recall | 21.9% | 88.0% |
| facts found in tables | 0.0% | 81.1% |
| support: span | 21.6% | 86.4% |
| support: multi-span | 22.2% | 91.2% |
| support: arithmetic | 16.9% | 82.7% |
| support: count | 25.0% | 78.1% |
| tables found | 0 | 573 |
| cells found | 0 | 16,699 |
| cells parsed as numbers | 0 | 9,193 |
| sections found | 48 | 808 |
| seconds per document | 0.2 | 37.4 |
| failed documents | 0 | 0 |

The deep-learning backend is worth its cost by a factor of four, and the
mechanism is visible rather than assumed: `textonly` recovers no table
structure at all, so 77.7% of everything it loses is lost for that one
reason, and arithmetic questions — the ones that need two figures out of a
grid — are where it collapses hardest.

37s per document on a CPU-only torch build is the number that plans a corpus
run: 2,758 documents is roughly 28 hours single-threaded, which is why the
cache exists and why the corpus is meant to be parsed once offline.

> These are the **Phase 2 baseline** numbers, frozen at
> `baselines/PHASE_2_BASELINE/`. A later phase adds a section below it; it
> never rewrites this table.
>
> `textonly` was the control this comparison needed and is no longer a backend
> the service offers. Its implementation was moved to
> `evaluation/textonly_baseline.py` rather than deleted, so this row can still
> be regenerated: `run_extraction_eval.py` registers it by name.

### The finding worth acting on

Docling's remaining weakness is not table detection, it is **table scale**:

| Gold says | correct | missed | wrong |
|---|---|---|---|
| thousand | 121 | 146 | 46 |
| million | 78 | 139 | 1 |
| billion | 0 | 4 | 0 |

**37.2%** accuracy on the questions whose gold declares a magnitude. `missed`
dominates: the parser found the table and the figures, and did not find the
"in thousands" that multiplies them. Every one of those is a 1000x error with
a correct-looking citation attached — the exact failure this service was
built to prevent.

The cause is in `app/processing/finance.py`: `parse_scale` reads only
`Table.caption`, and real filings state magnitude in a column header
(`$ in thousands`), in the row-label stem, or in the paragraph above the
table just as often as in a caption. Widening it is the highest-value change
available to this service, and this harness is how the fix gets proved rather
than asserted.

Before widening the detector, `scripts/inspect_scale_misses.py` asked the
corpus where the evidence actually sat, mining the Phase 2 cache for the
declared-magnitude questions the caption-only rule got wrong:

| Where the magnitude phrase was available | Cases |
|---|---:|
| header cell / flattened column header | ~150 |
| row-label stem (column zero) | ~108 |
| a block above the table | ~72 |
| **nowhere in the parse at all** | **87** |

Docling leaves `caption` empty on nearly every table in this corpus. The
detector was looking in the one place the words were not.

### Phase 2.1 — financial scale recovery

`app/processing/scale.py` resolves magnitude from four sources in precedence
order — **caption > header > row stem > preceding text** — recording the
winning source as provenance.

A magnitude word alone is not a declaration. Two anchored patterns are
recognised, `in <magnitude>` and `<currency> <magnitude>`, and both reject a
magnitude immediately preceded by a number — which is what separates the
declaration `(in millions)` from the figure `$5 million`. Preceding prose is
held to a stricter rule still: parenthesised, or carrying a declarative
lead-in, so "The following amounts are in thousands." is admitted while "we
operate in millions of households" is not. A conflict between *different*
sources is settled by precedence, deterministically; a conflict *inside* the
winning source is the case no rule can settle, so it is flagged `ambiguous`
and logged with every candidate rather than silently decided.

Same 274 documents, same 1,644 questions, same 2,804 gold facts, same matching
rules and metric definitions. The processor is the only changed variable:

| Metric | Phase 2 | Phase 2.1 | Δ |
|---|---:|---:|---:|
| **scale accuracy (gold declares one)** | **37.2%** | **78.9%** | **+41.7 pts** |
| scale accuracy (all scorable) | 51.8% | 74.3% | +22.5 pts |
| question support | 85.3% | 85.3% | 0 |
| fact recall | 88.0% | 88.0% | 0 |
| facts found in tables | 81.1% | 81.1% | 0 |
| tables / cells / cells parsed | 573 / 16,699 / 9,193 | 573 / 16,699 / 9,193 | 0 |
| seconds per document | 37.4 | 34.4 | −3.0 |

**223 of the 336 previously wrong magnitude questions recovered. Regressions: 0.**

The flat rows are the point, not an omission. Scale sets
`NumericValue.scaled` and never `num`, and fact matching compares `num` — so a
correct intervention *must* leave recall, support, and structure untouched.
All 2,804 fact records compare byte-identical between the two runs,
identifiers included.

`scripts/analyze_scale.py` classifies every remaining decision and diffs two
runs question by question:

```bash
python scripts/analyze_scale.py \
    --before baselines/PHASE_2_BASELINE/dev-docling.jsonl \
    --after  runs/dev21-docling.jsonl --examples 12
```

What is left is mostly a ceiling rather than a bug. On 87 of the
declared-magnitude questions no magnitude word exists anywhere in Docling's
parse — headers read `2019`/`2018`, the block above reads `Table of Contents`,
and the phrase the reader saw on the page is simply not in the extraction.
448 of 535 (83.7%) is the most any detector reading this parse can reach.
The strict false-positive signal — both sides declare a magnitude and they
differ — fell from 47 cases to 1, and that one is arguably not an error:
`microsoft-corporation_2019` prints `(In millions, except percentages and per
share amounts)`, which is what the detector read; gold says `billion` because
the *answer* is in billions.

Two limits worth stating. A table gets a single scale, so
`(In thousands, except per share amounts)` is read as thousands and the
per-share columns are not exempted — percentages are already safe, because
`parse_value` never scales them. And provenance is re-derived at analysis
time rather than stored: `ProcessedDocument` v1.0 carries `TableUnits.scale`,
`scale_label`, and `currency`, but not which source won. Every input the
detector reads is already a contract field, so `analyze_scale.py` reproduces
the decision exactly from a cached document. Carrying provenance at serving
time would be an additive `TableUnits` field and a conscious contract
decision, not a silent one.

`cache_variant` includes `PROCESSING_VERSION` for this reason: what the cache
stores is the *assembled* document, not the backend's raw parse, so a change
to scale detection produces a different document from identical bytes and
identical models. Without the version in the key, measuring this change would
have silently scored the old build. Phase 2 entries live under `-p1` and
remain on disk; Phase 2.1 writes `-p2`.

Scale is where docling's headline weakness was. The other 12% of its misses
have nothing to do with magnitude: `table_found_cell_missed` (71.5%) and
prose wording differences (28.5%).

### What is being scored, and against what

TAT-DQA ships no reference parse of its documents — only the questions. What
each question carries is `facts`: the literal strings a reader must find in
the document to answer it, plus the magnitude those figures are printed in.
That turns out to be the better target anyway. A cell recovered but never
asked about proves nothing, while **a fact the parser lost is one no amount of
retrieval or reasoning can recover.** So:

| Metric | Means |
|---|---|
| **fact recall** | Of every string some question needs, how many the parser recovered. Moves smoothly; the right signal when tuning. |
| **question support** | Of every question, how many had *all* their facts recovered. The ceiling on end-to-end accuracy — a question missing one operand is unanswerable, not half-answerable. |
| **table share** | Of recovered facts, how many came from a table cell rather than prose. The clearest single signal of table understanding. |
| **scale accuracy** | Whether the table's detected magnitude matches the gold. Reported twice: over everything, and over only the questions whose gold declares thousands/millions/billions — because half of TAT-DQA declares no scale, and a parser that never detects one would score those correct for free. |

Every fact resolves to exactly one outcome, best first: `table_cell_text`,
`table_cell_value` (same number, written differently — `(1,234)` against
`-1,234`), `block_text`, `block_text_value`, `furniture_only`, `not_found`.

`furniture_only` is counted as a **miss** and reported separately. Those
blocks are excluded from `reading_order`, so a fact found only there never
reaches a retrieval chunk. When it is not near zero, page-furniture demotion
is eating content.

### Where the number can flatter a parser

Stated here so nobody quotes it wrongly:

* It is **recall, not precision**. A bare `0.4` matches any cell holding 0.4.
  The report carries an *unambiguous* rate — the share of recovered facts that
  matched exactly one place — beside the headline, so the ambiguity is visible
  rather than buried.
* It says nothing about whether a figure was attached to the **right label**,
  only that it was recovered. Row/column correctness needs a reference grid,
  which this dataset does not provide.
* Short numeric facts are easier to hit than long prose ones, so scores are
  broken out by answer type.

None of that weakens the comparison the harness exists for: both backends are
scored by the identical rule, so the difference between them is real even
where the absolute number is generous.

### Why the misses happened

An aggregate that cannot say *why* is a scoreboard, not a tool. `--failures`
attributes every unrecovered fact to something observable in the parse, and
names the documents to open first:

```bash
python scripts/run_extraction_eval.py --compare runs/dev-textonly.jsonl --failures
```

```
  2189 unrecovered facts, by cause:
    no_tables_recovered         1700   77.7%  a figure was needed and the parser found no table
    prose_not_matched            487   22.2%  a phrase the extracted text does not contain
    demoted_to_furniture           2    0.1%  present, but excluded from reading order
```

That first line is the case for the deep-learning backend in one number:
three quarters of what the text-layer reader loses, it loses because it never
recovered a table at all.

The buckets are evidence, not verdicts — `prose_not_matched` includes wording
the extractor rendered differently (ligatures, hyphenation, spacing), not only
text that was lost.

### Running it

`--limit N` samples deterministically by document uid, so two backends score
the identical documents and the difference between them is the backend and
nothing else. Results stream to JSONL after every document, so a run killed at
document 200 of 274 keeps 200 usable results; `--resume` picks it up, and
refuses if the file was produced by a different backend, sample, or table
mode. Parses go through the service's own content-addressed cache, so
re-scoring with a changed rule costs seconds rather than hours — and the cache
a harness run leaves behind is the same one the demo serves from.

On CPU, Docling is around 30s per page, so the full dev split is a couple of
hours. `--limit 40` gives a usable comparison in twenty minutes.

---

## Reuse from `thorn-nlp`

Four modules were ported and stripped of their Arabic/legal specifics; the old
project is never imported, so none of its `torch==2.2.0` / `transformers==4.38.0`
pins or its two-virtualenv split come along.

| Ported | Change |
|---|---|
| `vision/pdf_render.py` | Kept its pypdfium2 approach — no Poppler binary to install. Extended with per-glyph font size and weight, which the Docling backend needs for heading levels |
| `reading_order/order.py` | Already had an `rtl` flag; passes `rtl=False` |
| `detect_column_count` | Geometric, taken as-is |
| Deterministic block ids | `p{page:04d}_b{index:03d}` scheme kept |

Also inherited, as stances rather than code: fail loud on misconfiguration,
isolate per-page failures so one bad page never costs the document, and inject
backends through a Protocol so tests need no GPU.

Deliberately **not** reused: everything Arabic-specific (`normalize_arabic`,
`_reverse_visual_arabic`, script routing, Qari/Kraken, the lexicon repairer)
and the hard adaptive-threshold binarisation, which destroys the thin table
rule lines a table-structure model depends on.
