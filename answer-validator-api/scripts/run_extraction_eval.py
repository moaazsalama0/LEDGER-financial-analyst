"""Score a backend's extraction against the TAT-DQA gold questions.

This is the Phase 2 harness the README and ``inspect_document.py`` point at:
the number that settles a backend choice, produced the same way every time so
two runs can be compared honestly.

    # the production backend
    python scripts/run_extraction_eval.py --backend docling --limit 50

    # the frozen Phase 2 control: no model, no GPU. Not a backend the service
    # serves any more — it is registered by this script alone, so the
    # historical arm stays reproducible without production offering it.
    python scripts/run_extraction_eval.py --backend textonly --limit 50

    # the artefact: both, side by side
    python scripts/run_extraction_eval.py --compare \\
        runs/dev-textonly.jsonl runs/dev-docling.jsonl

``--limit`` samples deterministically from the split, so the two runs above
score the identical documents and the difference between them is the backend
and nothing else. Results stream to JSONL as they are produced; ``--resume``
picks a killed run back up where it stopped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent)]

from app.core.settings import Settings  # noqa: E402
from evaluation import dataset, harness, report as reporting  # noqa: E402
from evaluation import textonly_baseline  # noqa: E402

# The Phase 2 control is not a backend the service offers; it is registered
# here so `--backend textonly` still reproduces the frozen historical run.
textonly_baseline.register()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", default="docling")
    parser.add_argument("--split", default="dev", choices=dataset.SPLITS)
    parser.add_argument("--data-dir", type=Path, default=Path("data/tatdqa"))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score a deterministic sample of this many documents.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sample seed.")
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Parse every document afresh instead of reusing the service cache.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Results JSONL. Defaults to runs/<split>-<backend>.jsonl.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Reuse results already in --out."
    )
    parser.add_argument("--json", type=Path, default=None, help="Also write the report as JSON.")
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="RESULTS.jsonl",
        default=None,
        help="Render existing result files side by side and exit.",
    )
    parser.add_argument(
        "--common",
        action="store_true",
        help="With --compare, score only the documents every run covers.",
    )
    parser.add_argument(
        "--failures",
        type=int,
        nargs="?",
        const=5,
        default=None,
        metavar="N",
        help="Also root-cause the unrecovered facts and name the N worst documents.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress the per-document progress line."
    )
    return parser.parse_args()


def do_compare(paths: list[Path], failures: int | None, common: bool) -> int:
    loaded = []
    for path in paths:
        meta, results = harness.read_results(path)
        if not results:
            print(f"{path}: no results", file=sys.stderr)
            return 1
        loaded.append((meta, results))

    runs = [results for _, results in loaded]
    if common and len(runs) > 1:
        shared = reporting.common_documents(runs)
        if not shared:
            print("the runs share no documents", file=sys.stderr)
            return 1
        loaded = [(meta, reporting.restrict(results, shared)) for meta, results in loaded]
        runs = [results for _, results in loaded]
        print(f"comparing the {len(shared)} documents every run covers\n")

    warning = reporting.coverage_warning(runs)

    reports = []
    for meta, results in loaded:
        reports.append(reporting.aggregate(results, meta))
        print(reporting.render(reports[-1]))
        print()
        if failures:
            print(reporting.render_failures(results, failures))
            print()

    print(reporting.compare(reports))
    if warning:
        print()
        print(warning)
    return 0


def main() -> int:
    args = parse_args()

    if args.compare:
        return do_compare([Path(p) for p in args.compare], args.failures, args.common)

    try:
        split = dataset.load_split(args.data_dir, args.split)
    except dataset.DatasetError as exc:
        print(exc, file=sys.stderr)
        return 2

    documents = dataset.sample(split.documents, limit=args.limit, seed=args.seed)
    if not documents:
        print("no documents selected", file=sys.stderr)
        return 2

    out = args.out or Path("runs") / f"{args.split}-{args.backend}.jsonl"
    version, models = harness.backend_meta(args.backend)
    meta = harness.RunMeta(
        backend=args.backend,
        backend_version=version,
        model_versions=models,
        split=args.split,
        seed=args.seed,
        limit=args.limit,
        documents=len(documents),
        force_ocr=args.force_ocr,
        started_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )

    questions = sum(len(document.questions) for document in documents)
    print(
        f"{args.split}: {len(documents)} documents, {questions} questions"
        f"  ->  {args.backend}",
        file=sys.stderr,
    )
    if split.missing_pdfs:
        print(
            f"  note: {len(split.missing_pdfs)} gold documents have no PDF and "
            "were skipped",
            file=sys.stderr,
        )

    def progress(position: int, total: int, result: harness.DocumentResult) -> None:
        if args.quiet:
            return
        state = "FAILED" if not result.ok else ("cached" if result.from_cache else f"{result.elapsed_s:.1f}s")
        supported = sum(1 for q in result.questions if q.supported)
        sys.stderr.write(
            f"\r  [{position:>4}/{total}] {result.document_uid[:12]}  "
            f"{state:>8}  {supported}/{len(result.questions)} questions supported     "
        )
        sys.stderr.flush()

    results = list(
        harness.run(
            documents,
            backend_name=args.backend,
            settings=Settings(),
            out=out,
            meta=meta,
            force_ocr=args.force_ocr,
            use_cache=not args.no_cache,
            resume=args.resume,
            on_result=progress,
        )
    )
    if not args.quiet:
        sys.stderr.write("\n\n")

    summary = reporting.aggregate(results, meta)
    print(reporting.render(summary))
    if args.failures:
        print()
        print(reporting.render_failures(results, args.failures))
    print(f"\nresults {out}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(reporting.to_json(summary), encoding="utf-8")
        print(f"report  {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
