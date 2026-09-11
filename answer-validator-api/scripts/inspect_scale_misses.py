"""Where does the magnitude text actually live in the documents we get wrong?

Phase 2 measured scale recovery at 37.2% on questions whose gold declares a
magnitude, with 'missed' dominating. Before widening the detector, this asks
the corpus where the evidence it should have used was sitting: in the
caption, in a header cell, in a row label, in the paragraph above, or nowhere
recoverable at all.

Runs entirely off the Phase 2 results file and the document cache, so it
costs seconds and re-parses nothing.

    python scripts/inspect_scale_misses.py --limit 25
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent)]

from app.backends import registry  # noqa: E402
from app.cache.store import DocumentCache, cache_variant  # noqa: E402
from app.core.settings import Settings  # noqa: E402
from app.ingestion.validate import content_sha256  # noqa: E402
from evaluation import dataset  # noqa: E402
from ledger_doc_contract.v1.models import ProcessedDocument, Table  # noqa: E402

# Any mention of a magnitude word, however it is phrased. Deliberately much
# looser than the production detector: the point is to find where the words
# are, not to decide what they mean.
MAGNITUDE = re.compile(r"\b(thousand|million|billion|trillion)s?\b", re.IGNORECASE)

DECLARED = {"thousand": 1e3, "million": 1e6, "billion": 1e9}


def load_document(pdf: Path, cache: DocumentCache, variant: str) -> ProcessedDocument | None:
    return cache.get(content_sha256(pdf.read_bytes()), variant)


def header_texts(table: Table) -> list[str]:
    """Every header cell's text, plus the flattened column headers."""
    out = [cell.text for cell in table.cells if cell.is_header and cell.text.strip()]
    out.extend(h for h in table.column_headers if h.strip())
    return out


def row_stem_texts(table: Table) -> list[str]:
    """Column-zero labels, which is where a row-label stem would sit."""
    return [
        cell.text
        for cell in table.cells
        if cell.col == 0 and cell.text.strip()
    ]


def body_cell_texts(table: Table) -> list[str]:
    return [
        cell.text
        for cell in table.cells
        if not cell.is_header and cell.col > 0 and cell.text.strip()
    ]


def preceding_blocks(document: ProcessedDocument, table: Table, window: int) -> list[str]:
    """Text of the ``window`` blocks before this table's block, in reading order."""
    page = document.page(table.page_number)
    if page is None:
        return []
    order = list(page.reading_order)
    if table.block_id not in order:
        order = [b.block_id for b in page.blocks]
    try:
        position = order.index(table.block_id)
    except ValueError:
        return []
    by_id = {b.block_id: b for b in page.blocks}
    out = []
    for block_id in order[max(0, position - window) : position]:
        block = by_id.get(block_id)
        if block is not None and block.text.strip():
            out.append(block.text)
    return out


def locate(document: ProcessedDocument, table: Table, window: int) -> dict[str, list[str]]:
    """Every place a magnitude word appears for this table, by location."""
    found: dict[str, list[str]] = {}

    def record(where: str, texts: list[str]) -> None:
        hits = [t for t in texts if MAGNITUDE.search(t)]
        if hits:
            found[where] = hits[:3]

    record("caption", [table.caption] if table.caption else [])
    record("header", header_texts(table))
    record("row_stem", row_stem_texts(table))
    record("body_cell", body_cell_texts(table))
    for distance in (1, 2, 3, 5, window):
        blocks = preceding_blocks(document, table, distance)
        hits = [t for t in blocks if MAGNITUDE.search(t)]
        if hits:
            found[f"preceding<={distance}"] = hits[:2]
            break
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("runs/dev-docling.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/tatdqa"))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--backend", default="docling")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--limit", type=int, default=20, help="Examples to print.")
    args = parser.parse_args()

    settings = Settings()
    backend = registry.get_backend(args.backend)
    variant = cache_variant(backend.name, backend.model_versions)
    cache = DocumentCache(settings.cache_dir, enabled=True)

    split = dataset.load_split(args.data_dir, args.split)
    gold_by_uid = {d.uid: d for d in split.documents}

    rows = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = [r for r in rows if r.get("record") == "document"]

    where: Counter = Counter()
    verdicts: Counter = Counter()
    printed = 0
    documents: dict[str, ProcessedDocument] = {}

    for result in results:
        if not result.get("ok"):
            continue
        gold = gold_by_uid.get(result["document_uid"])
        if gold is None:
            continue

        for question in result["questions"]:
            gold_scale = question["gold_scale"]
            verdict = question["scale_verdict"]
            if gold_scale not in DECLARED or verdict == "unscorable":
                continue
            verdicts[verdict] += 1
            if verdict == "correct":
                continue

            table_ids = [f["table_id"] for f in question["facts"] if f.get("table_id")]
            if not table_ids:
                continue

            if gold.uid not in documents:
                loaded = load_document(gold.pdf_path, cache, variant)
                if loaded is None:
                    continue
                documents[gold.uid] = loaded
            document = documents[gold.uid]

            table = document.table(table_ids[0])
            if table is None:
                continue

            places = locate(document, table, args.window)
            key = "|".join(sorted(places)) if places else "NOWHERE"
            where[key] += 1

            if printed < args.limit:
                printed += 1
                print("=" * 74)
                print(f"doc {gold.uid[:12]}  {gold.source}  page {table.page_number}")
                print(f"  gold scale     {gold_scale}   verdict {verdict}")
                print(f"  detected scale {table.units.scale:g}  label={table.units.scale_label!r}")
                print(f"  caption        {(table.caption or '')[:90]!r}")
                if not places:
                    print("  NO magnitude word anywhere in the table or its context")
                for place, hits in places.items():
                    for hit in hits:
                        print(f"  [{place}] {hit[:110]!r}")

    print()
    print("=" * 74)
    print("verdicts on declared-scale questions:", dict(verdicts))
    print()
    print("where the magnitude word WAS available, for non-correct questions:")
    for key, count in where.most_common(15):
        print(f"  {count:>5}  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
