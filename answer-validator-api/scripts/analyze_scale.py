"""Classify every magnitude decision, and diff two runs question by question.

The aggregate says whether Phase 2.1 helped. This says *how*, and where it
still does not: each scorable question is put in one of six buckets, with the
evidence the detector saw and the interpretation the answer would carry.

Provenance is re-derived rather than stored. Every input the detector reads —
caption, header cells, column-zero labels, the blocks above the table — is
already a field of ``ProcessedDocument``, so running the detector again over a
cached document reproduces its decision exactly, and the v1.0 contract does
not have to grow a field to carry an answer it already implies.

    python scripts/analyze_scale.py \\
        --before baselines/PHASE_2_BASELINE/dev-docling.jsonl \\
        --after  runs/dev21-docling.jsonl --examples 12
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent)]

from app.backends import registry  # noqa: E402
from app.cache.store import DocumentCache, cache_variant  # noqa: E402
from app.core.settings import Settings  # noqa: E402
from app.ingestion.validate import content_sha256  # noqa: E402
from app.processing.scale import detect_scale  # noqa: E402
from evaluation import dataset  # noqa: E402
from ledger_doc_contract.v1.models import ProcessedDocument, Table  # noqa: E402

# The six buckets the Phase 2.1 brief asks for.
BUCKETS: dict[str, str] = {
    "A_correct": "scale correctly recovered",
    "B_missed": "scale still missed (gold declares one, none found)",
    "C_wrong": "wrong scale recovered (both declare, they differ)",
    "D_ambiguous": "conflicting declarations inside the winning source",
    "E_spurious": "unrelated text read as a scale (gold declares none)",
    "F_numeric": "numeric extraction failure, unrelated to scale",
}

DECLARED = {"thousand": 1e3, "million": 1e6, "billion": 1e9}


def load_results(path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("record") == "run":
            meta = payload
        elif payload.get("record") == "document":
            rows.append(payload)
    return meta, rows


def table_context(document: ProcessedDocument, table: Table, window: int = 5):
    """The four inputs the detector reads for this table."""
    headers = [c.text for c in table.cells if c.is_header and c.text.strip()]
    headers += [h for h in table.column_headers if h.strip()]
    row_stems = [c.text for c in table.cells if c.col == 0 and c.text.strip()]

    preceding: list[str] = []
    page = document.page(table.page_number)
    if page is not None:
        order = list(page.reading_order) or [b.block_id for b in page.blocks]
        by_id = {b.block_id: b for b in page.blocks}
        if table.block_id in order:
            position = order.index(table.block_id)
            for block_id in reversed(order[max(0, position - window) : position]):
                block = by_id.get(block_id)
                if block is not None and block.text.strip():
                    preceding.append(block.text.strip())
    return table.caption, headers, row_stems, preceding


def classify(question: dict, table: Table | None, ambiguous: bool) -> str:
    verdict = question["scale_verdict"]
    if verdict == "correct":
        return "D_ambiguous" if ambiguous else "A_correct"
    if verdict == "missed":
        return "D_ambiguous" if ambiguous else "B_missed"
    if verdict == "wrong":
        return "D_ambiguous" if ambiguous else "C_wrong"
    if verdict == "spurious":
        return "E_spurious"
    # unscorable: no table backed it. That is a retrieval/extraction miss, not
    # a scale decision, and is only counted when the facts were actually lost.
    if any(not f["outcome"].startswith(("table_", "block_")) for f in question["facts"]):
        return "F_numeric"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=None)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/tatdqa"))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--backend", default="docling")
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    settings = Settings()
    backend = registry.get_backend(args.backend)
    variant = cache_variant(backend.name, backend.model_versions)
    cache = DocumentCache(settings.cache_dir, enabled=True)

    split = dataset.load_split(args.data_dir, args.split)
    gold_by_uid = {d.uid: d for d in split.documents}

    _, after_rows = load_results(args.after)
    before_verdicts: dict[str, str] = {}
    if args.before:
        _, before_rows = load_results(args.before)
        for row in before_rows:
            for question in row["questions"]:
                before_verdicts[question["uid"]] = question["scale_verdict"]

    # Does gold itself corroborate a 'spurious' detection? If another question
    # on the SAME table declares the magnitude we detected, the table really is
    # scaled and the spurious verdict is only the metric comparing the answer's
    # magnitude against the table's. This is the evidence that separates a
    # metric artifact from a real false positive, and it needs no human.
    gold_scales_by_table: dict[tuple[str, str], set[float]] = {}
    for row in after_rows:
        if not row.get("ok"):
            continue
        for question in row["questions"]:
            factor = DECLARED.get(question["gold_scale"])
            if factor is None:
                continue
            for fact in question["facts"]:
                if fact.get("table_id"):
                    key = (row["document_uid"], fact["table_id"])
                    gold_scales_by_table.setdefault(key, set()).add(factor)

    corroboration: Counter = Counter()

    buckets: Counter = Counter()
    transitions: Counter = Counter()
    # Where a 'spurious' detection came from. The frozen metric calls a
    # detection spurious whenever gold declares no magnitude, but gold's empty
    # scale means "the ANSWER carries no magnitude" — 679 of the dev split's
    # 781 such questions are text spans or counts — not "the TABLE is
    # unscaled". A millions table can host a question answered by a share
    # count. So the bucket is split by provenance: a phrase read from inside
    # the table is almost certainly a real declaration that simply does not
    # apply to this answer, while one read from surrounding prose is the case
    # worth suspecting.
    spurious_source: Counter = Counter()
    spurious_label: Counter = Counter()
    examples: dict[str, list[str]] = {name: [] for name in BUCKETS}
    documents: dict[str, ProcessedDocument] = {}

    for row in after_rows:
        if not row.get("ok"):
            continue
        gold = gold_by_uid.get(row["document_uid"])
        if gold is None:
            continue

        for question in row["questions"]:
            gold_scale = question["gold_scale"]
            verdict = question["scale_verdict"]

            previous = before_verdicts.get(question["uid"])
            if previous is not None and gold_scale in DECLARED:
                transitions[(previous, verdict)] += 1

            table_ids = [f["table_id"] for f in question["facts"] if f.get("table_id")]
            table = None
            ambiguous = False
            detection = None
            if table_ids:
                if gold.uid not in documents:
                    loaded = cache.get(content_sha256(gold.pdf_path.read_bytes()), variant)
                    if loaded is None:
                        continue
                    documents[gold.uid] = loaded
                document = documents[gold.uid]
                table = document.table(table_ids[0])
                if table is not None:
                    caption, headers, stems, preceding = table_context(document, table)
                    detection = detect_scale(
                        caption=caption,
                        headers=headers,
                        row_stems=stems,
                        preceding=preceding,
                    )
                    ambiguous = detection.ambiguous

            bucket = classify(question, table, ambiguous)
            if not bucket:
                continue
            buckets[bucket] += 1
            if bucket == "E_spurious" and table is not None:
                declared = gold_scales_by_table.get(
                    (row["document_uid"], table.table_id), set()
                )
                if table.units.scale in declared:
                    corroboration["confirmed_by_gold_on_the_same_table"] += 1
                elif declared:
                    corroboration["gold_declares_a_DIFFERENT_magnitude"] += 1
                else:
                    corroboration["no_gold_evidence_either_way"] += 1
            if bucket == "E_spurious" and detection is not None:
                spurious_source[
                    detection.source.value if detection.source else "none"
                ] += 1
                if table is not None and table.units.scale_label:
                    spurious_label[table.units.scale_label.lower()] += 1

            if len(examples[bucket]) >= args.examples or table is None:
                continue

            fact = next(
                (f for f in question["facts"] if f.get("cell_id")), question["facts"][0]
            )
            cell = next(
                (c for c in table.cells if c.cell_id == fact.get("cell_id")), None
            )
            printed = cell.text if cell else fact["fact"]
            num = cell.value.num if cell and cell.value else None
            expected = DECLARED.get(gold_scale, 1.0)
            got = table.units.scale

            lines = [
                f"  document   {gold.uid[:12]}  {gold.source}",
                f"  page       {table.page_number}   table {table.table_id}",
                f"  question   {gold_scale!r} expected, detector said {got:g}",
                f"  detected   {got:g}"
                + (
                    f" from {detection.source.value} ({table.units.scale_label!r})"
                    if detection and detection.source
                    else " (nothing found)"
                ),
                f"  expected   {expected:g}  (gold scale {gold_scale!r})",
                f"  printed    {printed!r}   parsed num {num}",
            ]
            if num is not None:
                lines.append(
                    f"  reading    got {num * got:,.0f}   should be {num * expected:,.0f}"
                )
            caption, headers, stems, preceding = table_context(document, table)
            evidence = (
                [f"caption:{caption!r}"] if caption else []
            ) + [f"header:{h!r}" for h in headers[:2]] + [
                f"row_stem:{s!r}" for s in stems[:1]
            ] + [f"preceding:{p[:60]!r}" for p in preceding[:1]]
            lines.append("  evidence   " + " | ".join(evidence[:4]))
            examples[bucket].append("\n".join(lines))

    print("=" * 74)
    print("SCALE ERROR CLASSIFICATION")
    print("=" * 74)
    total = sum(buckets.values())
    for name, description in BUCKETS.items():
        count = buckets[name]
        share = 100.0 * count / total if total else 0.0
        print(f"  {name:<14}{count:>6}  {share:5.1f}%   {description}")
    print(f"  {'total':<14}{total:>6}")

    if spurious_source:
        print()
        print("=" * 74)
        print("E_spurious by provenance")
        print("=" * 74)
        print("  gold's empty scale means the ANSWER carries no magnitude, not")
        print("  that the table is unscaled, so a declaration read from inside")
        print("  the table is most likely real but irrelevant to that answer.")
        for source, count in spurious_source.most_common():
            print(f"    {source:<16}{count:>6}")
        print("  corroboration from gold:")
        for name, count in corroboration.most_common():
            note = ""
            if name.startswith("confirmed"):
                note = "   <- not a false positive"
            elif name.startswith("gold_declares_a_DIFF"):
                note = "   <- a real false positive"
            print(f"    {name:<44}{count:>5}{note}")
        print("  most common phrases read:")
        for label, count in spurious_label.most_common(8):
            print(f"    {count:>5}  {label!r}")

    if transitions:
        print()
        print("=" * 74)
        print("BEFORE -> AFTER on declared-magnitude questions")
        print("=" * 74)
        recovered = 0
        regressed = 0
        for (before, after), count in sorted(transitions.items(), key=lambda kv: -kv[1]):
            arrow = f"{before:>9} -> {after:<9}"
            note = ""
            if before != "correct" and after == "correct":
                recovered += count
                note = "  RECOVERED"
            elif before == "correct" and after != "correct":
                regressed += count
                note = "  REGRESSED"
            print(f"  {arrow}{count:>6}{note}")
        print()
        print(f"  recovered {recovered}   regressed {regressed}")

    for name, description in BUCKETS.items():
        if not examples[name]:
            continue
        print()
        print("=" * 74)
        print(f"{name}  -  {description}")
        print("=" * 74)
        for block in examples[name]:
            print(block)
            print("  " + "-" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
