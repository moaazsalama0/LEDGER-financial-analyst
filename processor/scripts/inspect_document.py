"""Parse one PDF and print what came out of it.

The quality harness scored against the TAT-DQA gold JSON is the number that
settles a backend choice, and it is Phase 2 work. This is the thing you reach
for before that exists and long after: a single command that says what the
parser actually found in a document, so a wrong answer can be traced back to a
missing table or a flattened section tree rather than guessed at.

    python scripts/inspect_document.py report.pdf --backend docling
    python scripts/inspect_document.py report.pdf --backend docling --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent)]

from app.backends import registry  # noqa: E402
from app.core.settings import Settings  # noqa: E402
from app.ingestion.validate import content_sha256  # noqa: E402
from app.processing.assemble import assemble  # noqa: E402
from evaluation import textonly_baseline  # noqa: E402
from ledger_doc_contract.v1.models import ProcessedDocument  # noqa: E402

# `--backend textonly` names the frozen Phase 2 control, which the service no
# longer serves. Registered here so a historical parse can still be inspected.
textonly_baseline.register()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--backend", default="docling")
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument(
        "--json", type=Path, default=None, help="Also write the full document here."
    )
    return parser.parse_args()


def summarise(document: ProcessedDocument, elapsed: float) -> None:
    types = Counter(b.type.value for b in document.iter_blocks())
    numeric = sum(
        1 for t in document.tables for c in t.cells if c.value and c.value.num is not None
    )
    cells = sum(len(t.cells) for t in document.tables)

    print(f"document      {document.document_id}  ({document.filename})")
    print(f"backend       {document.processing.backend} {document.processing.backend_version}")
    for name, model in document.processing.model_versions.items():
        print(f"  {name:<10}  {model}")
    print(f"pages         {document.page_count}")
    print(f"elapsed       {elapsed:.1f}s  ({elapsed / max(document.page_count, 1):.1f}s/page)")
    print()

    # iter_blocks() walks the reading order, which excludes page furniture.
    # Counting furniture separately rather than omitting it: "140 blocks"
    # beside a document that emitted 151 invites exactly the wrong conclusion.
    furniture = [
        b for p in document.pages for b in p.blocks if b.type.is_page_furniture
    ]
    print(f"blocks        {sum(types.values())}  (in reading order)")
    for kind, count in types.most_common():
        print(f"  {kind:<16}{count}")
    if furniture:
        kinds = Counter(b.type.value for b in furniture)
        print(f"  furniture       {len(furniture)}  {dict(kinds)} — emitted, not chunked")
        for text in sorted({b.text.strip()[:40] for b in furniture})[:6]:
            print(f"                    {text!r}")
    print()

    print(f"tables        {len(document.tables)}")
    print(f"  cells       {cells}")
    # The metric that predicts arithmetic-question success: a cell whose
    # figure was recovered *and* parsed is one the agent can compute with.
    print(f"  parsed nums {numeric}")
    for table in document.tables[:5]:
        location = document.section_path(table.section_id) or "(no section)"
        print(
            f"  {table.table_id}  p{table.page_number}  "
            f"{table.n_rows}x{table.n_cols}  scale={table.units.scale:g}  {location}"
        )
    print()

    print(f"sections      {len(document.sections)}")
    for section in document.sections[:12]:
        print(
            f"  {'  ' * (section.level - 1)}L{section.level} "
            f"p{section.page_start}-{section.page_end}  {section.title[:60]}"
        )
    print()

    source = Counter(p.extraction_source.value for p in document.pages)
    print(f"extraction    {dict(source)}")

    if document.processing.warnings:
        print(f"\nwarnings      {len(document.processing.warnings)}")
        for warning in document.processing.warnings:
            page = f"p{warning.page}" if warning.page else "-"
            print(f"  [{page}] {warning.code.value}: {warning.message}")


def main() -> int:
    args = parse_args()
    data = args.pdf.read_bytes()

    backend = registry.get_backend(args.backend)
    backend.warm()

    started = time.perf_counter()
    raw = backend.parse(data, force_ocr=args.force_ocr)
    elapsed = time.perf_counter() - started

    document = assemble(
        raw,
        document_id=f"doc_{content_sha256(data)[:16]}",
        filename=args.pdf.name,
        content_sha256=content_sha256(data),
        backend_name=backend.name,
        backend_version=backend.version,
        model_versions=backend.model_versions,
        duration_ms=int(elapsed * 1000),
        settings=Settings(),
    )

    summarise(document, elapsed)

    if args.json:
        args.json.write_text(
            json.dumps(document.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
