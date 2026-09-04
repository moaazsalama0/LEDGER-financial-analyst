# doc-processor-api — phase record

The measurement history of this service. Each phase records what was measured,
what changed, and what the change was worth. **Historical results are never
rewritten**; a later phase adds a section, it does not edit an earlier one.

Every number here comes from `scripts/run_extraction_eval.py` over the TAT-DQA
dev split, scored by `evaluation/matching.py`. The rule is the same in every
phase, so the rows are comparable.

---

## Phase 2 — baseline (FROZEN)

**Date:** 2026-09-04
**Frozen at:** `baselines/PHASE_2_BASELINE/`
**Dataset:** TAT-DQA dev — 274 documents, 307 pages, 1,644 questions, 2,804 gold facts
**Gold sha256:** `07395642f317754ed22bd8330525af7c1e15b7189efdf046cd0a257fc3178fcb`
**Processing version:** 1

Built the extraction quality harness and used it to settle the backend choice.

| | `textonly` | `docling` |
|---|---:|---:|
| question support | 19.7% | **85.3%** |
| fact recall | 21.9% | 88.0% |
| facts found in tables | 0.0% | 81.1% |
| support: span | 21.6% | 86.4% |
| support: multi-span | 22.2% | 91.2% |
| support: arithmetic | 16.9% | 82.7% |
| support: count | 25.0% | 78.1% |
| scale accuracy (all scorable) | n/a | 51.8% |
| **scale accuracy (gold declares one)** | n/a | **37.2%** |
| tables found | 0 | 573 |
| cells found | 0 | 16,699 |
| cells parsed as numbers | 0 | 9,193 |
| sections found | 48 | 808 |
| seconds per document | 0.2 | 37.4 |
| failed documents | 0 | 0 |

**Conclusion:** the deep-learning backend earns its cost by roughly four
times. 77.7% of everything `textonly` loses is lost because it recovers no
table structure at all.

**Weakness found:** table magnitude. On the 535 questions whose gold declares
thousands, millions, or billions, only 199 were right.

| Gold says | correct | missed | wrong |
|---|---:|---:|---:|
| thousand | 121 | 146 | 46 |
| million | 78 | 139 | 1 |
| billion | 0 | 4 | 0 |

`missed` dominated: the parser found the table and the figures, and did not
find the "in thousands" that multiplies them. Every one is a 1000x error
carrying a correct-looking citation.

---

## Phase 2.1 — financial scale recovery

**Date:** 2026-09-04
**Processing version:** 2
**Changed variable:** the processor only. Same dataset, same document subset,
same gold, same matching rules, same metric definitions.

### The diagnosis that drove it

`parse_scale` read magnitude from `Table.caption` alone. Mining the Phase 2
cache for where the words actually sat, across the declared-magnitude
questions the caption-only detector got wrong:

| Where the magnitude phrase was available | Cases |
|---|---:|
| header cell / flattened column header | ~150 |
| row-label stem (column zero) | ~108 |
| a block above the table | ~72 |
| **nowhere in the parse at all** | **87** |

Docling leaves `caption` empty on nearly every table in this corpus. The
detector was looking in the one place the words were not. The 87 "nowhere"
cases are an irreducible ceiling: at best 448 of 535 declared-magnitude
questions (83.7%) are recoverable by any detector reading this parse.

### The intervention

New module `app/processing/scale.py`. Magnitude is resolved from four sources
in precedence order — **caption > header > row stem > preceding text** — with
the winning source recorded as provenance.

A magnitude word alone is not a declaration. Two anchored patterns are
recognised, `in <magnitude>` and `<currency> <magnitude>`, and both reject a
magnitude immediately preceded by a number, which is what separates the
declaration `(in millions)` from the figure `$5 million`. Preceding prose is
held to a stricter rule still: parenthesised, or carrying a declarative
lead-in, so "The following amounts are in thousands." is admitted while "we
operate in millions of households" is not.

Conflicts between *different* sources are settled by precedence, deterministically.
A conflict *inside* the winning source is the case no rule can settle, so it is
flagged `ambiguous` and logged with every candidate rather than silently decided.

### Results

Same 274 documents, same 1,644 questions, same 2,804 gold facts, same matching
rules, same metric definitions. The processor is the only changed variable.

| Metric | Phase 2 | Phase 2.1 | Δ |
|---|---:|---:|---:|
| **scale accuracy (gold declares one)** | **37.2%** | **78.9%** | **+41.7 pts** |
| scale accuracy (all scorable) | 51.8% | 74.3% | +22.5 pts |
| question support | 85.3% | 85.3% | 0 |
| fact recall | 88.0% | 88.0% | 0 |
| facts found in tables | 81.1% | 81.1% | 0 |
| support: span | 86.4% | 86.4% | 0 |
| support: multi-span | 91.2% | 91.2% | 0 |
| support: arithmetic | 82.7% | 82.7% | 0 |
| support: count | 78.1% | 78.1% | 0 |
| unambiguous match rate | 78.1% | 78.1% | 0 |
| tables found | 573 | 573 | 0 |
| cells found | 16,699 | 16,699 | 0 |
| cells parsed as numbers | 9,193 | 9,193 | 0 |
| sections found | 808 | 808 | 0 |
| seconds per document | 37.4 | 34.4 | -3.0 |
| failed documents | 0 | 0 | 0 |

The flat rows are the point, not an omission. Scale changes `NumericValue.scaled`,
never `num`, and fact matching compares `num` — so a correct intervention
*must* leave recall, support, and structure untouched. All 2,804 fact records
compare byte-identical between the two runs, identifiers included.

**Magnitude questions recovered: 223 of 336 previously wrong. Regressions: 0.**

| Before → after | Count |
|---|---:|
| correct → correct | 199 |
| missed → correct | 187 · recovered |
| wrong → correct | 36 · recovered |
| missed → missed | 101 |
| wrong → missed | 10 |
| missed → wrong | 1 |
| wrong → wrong | 1 |
| unscorable → unscorable | 63 |

### Error classification

1,057 scorable decisions:

| Bucket | Count | Share | Meaning |
|---|---:|---:|---|
| A correct | 629 | 59.5% | scale correctly recovered |
| B missed | 111 | 10.5% | gold declares one, none found |
| C wrong | 1 | 0.1% | both declare, they differ |
| D ambiguous | 7 | 0.7% | conflicting declarations in the winning source |
| E spurious | 107 | 10.1% | gold declares none, we detected one |
| F numeric | 202 | 19.1% | fact lost before scale mattered |

**B (111) is the ceiling, not a bug.** Phase 2's mining found 87
declared-magnitude questions where no magnitude word exists anywhere in
Docling's parse. `greensky-inc_2019` is typical: headers read `2019`/`2018`,
the preceding block reads `Table of Contents`, and the phrase the reader saw
on the page is simply not in the extraction. No detector reading this parse
can recover those; fixing them means changing extraction upstream.

**C is a single case, and it is arguably not an error.**
`microsoft-corporation_2019` prints `(In millions, except percentages and per
share amounts)`; the detector read millions, which is what the table says.
Gold says `billion` because that *answer* is in billions.

**E is dominated by the same conflation.** 81 of 107 are confirmed by gold
itself — another question on the *same table* declares the magnitude we
detected, so the table really is scaled and only the answer carries none. 26
have no gold evidence either way. **Zero cases where gold declares a different
magnitude than the one detected**, which is the only signal that would
indicate a true false positive.

**False-positive verdict: none found.** The strict signal (`C`, both sides
declare and differ) fell from 47 in Phase 2 to 1.

### Verifying it by hand

See [`TESTING_SCALE.md`](TESTING_SCALE.md) — six levels from a 3-second unit
run to a full 2.5-hour reproduction, with the expected output for each.

### Known limitations

1. **The 87-case ceiling.** Where Docling's parse contains no magnitude word
   anywhere, no detector reading that parse can recover one. Fixing those
   needs a change upstream in extraction, not in this module.
2. **`spurious` is not a false-positive rate.** The frozen metric compares the
   *table's* detected scale against the *question's* gold scale, but gold's
   empty scale means "the answer carries no magnitude" — 679 of the dev
   split's 781 such questions are answered by a text span or a count — not
   "the table is unscaled". A millions table legitimately hosts a question
   answered by a share count. The metric was not changed, because the Phase 2
   baseline is frozen and comparability matters more; the `E_spurious` bucket
   in `scripts/analyze_scale.py` is therefore split by provenance instead.
   The clean false-positive signal is `wrong`, where both sides declare a
   magnitude and they differ.
3. **Provenance is re-derived, not stored.** `ProcessedDocument` v1.0 carries
   `TableUnits.scale`, `scale_label`, and `currency`, but not which source the
   phrase came from or whether candidates conflicted. Every input the detector
   reads is already a contract field, so `scripts/analyze_scale.py` reproduces
   the decision exactly by re-running the detector over a cached document. The
   contract was left unchanged deliberately. If a consumer ever needs
   provenance at serving time rather than at analysis time, that is an
   additive `TableUnits` field and a conscious contract decision, not a
   silent one.
4. **A single scale per table.** `(In thousands, except per share amounts)` is
   read as thousands; the per-share columns are not exempted. Percentages are
   already safe — `parse_value` never scales them — but a per-share column in
   a scaled table is still multiplied.

### Cache note

`cache_variant` now includes `PROCESSING_VERSION`. What the cache stores is
the *assembled* document, not the backend's raw parse, so a change to scale
detection produces a different document from identical bytes and identical
models. Without the version in the key, measuring this change would have
silently scored the old build. Phase 2 entries live under `-p1` and remain on
disk; Phase 2.1 writes `-p2`.
---

## Phase 2.2 — production hardening: one backend

**Date:** 2026-09-04
**Processing version:** 2 (unchanged)
**Measured variable:** none. No number in this file moves, and none was rerun.

Phase 2 needed two backends because a comparison needs a control. Phase 2.1
settled the remaining extraction question. What was left was a service
advertising a backend nobody should select: `textonly` recovered no table
structure at all, which is exactly what made it the right control and exactly
what makes it the wrong thing to serve.

### What changed

* `app/backends/registry.py` registers **Docling alone**. `GET /version`
  reports `available_backends: ["docling"]`, and `DOC_BACKEND` now defaults to
  `docling`.
* `POST /process` with `{"backend": "textonly"}` is refused with the existing
  `BACKEND_UNAVAILABLE` error contract — `503`, `requested`, and `available` —
  rather than quietly parsed by Docling. No new error schema.
* `registry.reset()` now restores the shipped registrations rather than only
  dropping instances, so a backend added by a tool or a test cannot leak into
  a later check of what the service offers.
* The service's operational port moved **8001 → 8008**.

### What was preserved, and why

The text-layer reader was **moved, not deleted** — from
`app/backends/textonly_backend.py` to `evaluation/textonly_baseline.py`. It is
no longer a backend the service knows about, and two jobs still needed it:

1. **Phase 2 stays reproducible.** `baselines/PHASE_2_BASELINE/dev-textonly.jsonl`
   is a frozen record, and a record nobody can regenerate is an assertion.
   `scripts/run_extraction_eval.py` registers the baseline by name, so
   `--backend textonly` reruns the historical arm exactly as it ran. `name`
   and the `textonly` cache variant are unchanged, so a rerun's provenance
   lines up with the frozen records.
2. **The fast suite stays model-free.** It parses a real PDF into a real
   `RawParse` on CPU in milliseconds, which is what lets validation,
   assembly, the contract, HTTP, and the cache be exercised on every commit.
   The test harness registers it; the service never does.

Everything else historical is untouched: the Phase 2 table above, the frozen
baselines, `runs/dev-textonly.*`, and the `textonly-p1` / `textonly-p2` cache
variants, which stay addressable under their own directories and cannot be
mistaken for a Docling parse.

### Contract

`ProcessedDocument` v1.0 is unchanged — no field, enum, or version moved.
`contract_version` is still `1.0`, and `contracts/.../v1/schema.json` is
byte-identical.
