"""Running a backend over a gold split and recording what it recovered.

The unit of work is one document: parse it, then score every question asked
about it against the parse. Parsing dominates the cost — on CPU, Docling is
minutes per document — so three properties matter more than they normally
would for an evaluation script.

**Reuse the service's own cache.** A parse is keyed on the PDF bytes and the
backend's model configuration, exactly as ``/process`` keys it. Re-scoring a
corpus with a changed matching rule then costs seconds rather than hours, and
the cache a harness run leaves behind is the same one the demo serves from.

**Write as you go.** Results are appended to JSONL after each document, so a
run interrupted at document 200 of 274 has 200 usable results rather than
none, and re-running skips what is already recorded.

**Never let one document end the run.** A parse failure is recorded against
that document and the run continues, because "docling failed on 3 of 274" is
a finding, and a traceback that loses the other 271 is not.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Sequence

from app.backends import registry
from app.backends.base import LayoutBackend
from app.cache.store import DocumentCache, cache_variant
from app.core.settings import Settings
from app.ingestion.validate import content_sha256
from app.processing.assemble import assemble
from evaluation.dataset import GoldDocument, GoldQuestion
from evaluation.matching import (
    FactMatch,
    Outcome,
    ScaleVerdict,
    build_index,
    match_fact,
    score_scale,
)
from ledger_doc_contract.v1.models import ProcessedDocument

SCHEMA_VERSION = 1
"""Bumped when a record's shape changes, so old result files are not silently
aggregated under new field meanings."""


@dataclass(frozen=True)
class QuestionResult:
    """How well the parse supports one gold question."""

    uid: str
    answer_type: str
    gold_scale: str
    facts: tuple[FactMatch, ...]
    scale_verdict: ScaleVerdict

    @property
    def supported(self) -> bool:
        """Every fact this question needs was recovered.

        The strict reading is the useful one: an arithmetic question missing
        one of its two operands is not 'half answerable', it is unanswerable.
        """
        return bool(self.facts) and all(fact.found for fact in self.facts)

    @property
    def found_count(self) -> int:
        return sum(1 for fact in self.facts if fact.found)


@dataclass(frozen=True)
class DocumentStructure:
    """What the parser found, independent of any question.

    Reported beside the gold-scored numbers because it explains them: a
    backend that recovers no tables will show it here before the fact recall
    makes it obvious.
    """

    page_count: int = 0
    blocks: int = 0
    furniture: int = 0
    sections: int = 0
    tables: int = 0
    cells: int = 0
    parsed_nums: int = 0
    scaled_tables: int = 0
    extraction_sources: dict[str, int] = field(default_factory=dict)
    warnings: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentResult:
    """One document's contribution to a run."""

    document_uid: str
    source: str
    ok: bool
    elapsed_s: float
    from_cache: bool = False
    structure: DocumentStructure = field(default_factory=DocumentStructure)
    questions: tuple[QuestionResult, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class RunMeta:
    """Everything needed to say what produced a set of results.

    A quality number without the configuration that produced it cannot settle
    an argument, so this is written into the results file itself rather than
    left in a shell history.
    """

    backend: str
    backend_version: str | None
    model_versions: dict[str, str]
    split: str
    seed: int
    limit: int | None
    documents: int
    force_ocr: bool
    gold_sha256: str = ""
    schema_version: int = SCHEMA_VERSION
    started_at: str = ""


def describe_structure(document: ProcessedDocument) -> DocumentStructure:
    """Summarise a parse without reference to any gold data."""
    sources: dict[str, int] = {}
    for page in document.pages:
        key = page.extraction_source.value
        sources[key] = sources.get(key, 0) + 1

    warnings: dict[str, int] = {}
    for warning in document.processing.warnings:
        key = warning.code.value
        warnings[key] = warnings.get(key, 0) + 1

    in_order = sum(1 for _ in document.iter_blocks())
    all_blocks = sum(len(page.blocks) for page in document.pages)

    return DocumentStructure(
        page_count=document.page_count,
        blocks=in_order,
        furniture=all_blocks - in_order,
        sections=len(document.sections),
        tables=len(document.tables),
        cells=sum(len(table.cells) for table in document.tables),
        parsed_nums=sum(
            1
            for table in document.tables
            for cell in table.cells
            if cell.value is not None and cell.value.num is not None
        ),
        scaled_tables=sum(1 for table in document.tables if table.units.scale != 1.0),
        extraction_sources=sources,
        warnings=warnings,
    )


def score_question(
    question: GoldQuestion,
    index,
    document: ProcessedDocument,
) -> QuestionResult:
    """Match every fact a question needs, then judge the table's scale."""
    matches = tuple(match_fact(fact, index) for fact in question.facts)

    # Only the tables this question's own facts landed in are relevant. A
    # document's other tables may legitimately be scaled differently.
    scales: list[float] = []
    for match in matches:
        if match.table_id is None:
            continue
        table = document.table(match.table_id)
        if table is not None:
            scales.append(table.units.scale)

    return QuestionResult(
        uid=question.uid,
        answer_type=question.answer_type,
        gold_scale=question.scale,
        facts=matches,
        scale_verdict=score_scale(
            question.scale, question.scale_factor, tuple(scales)
        ),
    )


def parse_document(
    pdf: Path,
    backend: LayoutBackend,
    settings: Settings,
    *,
    cache: DocumentCache | None = None,
    force_ocr: bool = False,
) -> tuple[ProcessedDocument, float, bool]:
    """Parse one PDF into the contract, serving from cache when possible.

    Returns the document, the seconds spent, and whether the cache answered.
    """
    data = pdf.read_bytes()
    digest = content_sha256(data)
    variant = cache_variant(backend.name, backend.model_versions)

    if cache is not None:
        cached = cache.get(digest, variant)
        if cached is not None:
            return cached, 0.0, True

    started = time.perf_counter()
    raw = backend.parse(data, force_ocr=force_ocr)
    elapsed = time.perf_counter() - started

    document = assemble(
        raw,
        document_id=f"doc_{digest[:16]}",
        filename=pdf.name,
        content_sha256=digest,
        backend_name=backend.name,
        backend_version=backend.version,
        model_versions=backend.model_versions,
        duration_ms=int(elapsed * 1000),
        settings=settings,
    )

    if cache is not None:
        cache.put(document, variant)

    return document, elapsed, False


def evaluate_document(
    gold: GoldDocument,
    backend: LayoutBackend,
    settings: Settings,
    *,
    cache: DocumentCache | None = None,
    force_ocr: bool = False,
) -> DocumentResult:
    """Parse one gold document and score every question about it."""
    try:
        document, elapsed, from_cache = parse_document(
            gold.pdf_path, backend, settings, cache=cache, force_ocr=force_ocr
        )
    except Exception as exc:  # a bad document must not end the run
        return DocumentResult(
            document_uid=gold.uid,
            source=gold.source,
            ok=False,
            elapsed_s=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )

    index = build_index(document)
    return DocumentResult(
        document_uid=gold.uid,
        source=gold.source,
        ok=True,
        elapsed_s=elapsed,
        from_cache=from_cache,
        structure=describe_structure(document),
        questions=tuple(
            score_question(question, index, document) for question in gold.questions
        ),
    )


# -- persistence ------------------------------------------------------------


def _encode(result: DocumentResult) -> dict:
    payload = asdict(result)
    payload["record"] = "document"
    for question in payload["questions"]:
        question["scale_verdict"] = ScaleVerdict(question["scale_verdict"]).value
        for fact in question["facts"]:
            fact["outcome"] = Outcome(fact["outcome"]).value
    return payload


def _decode(payload: dict) -> DocumentResult:
    structure = DocumentStructure(**payload.get("structure", {}))
    questions = tuple(
        QuestionResult(
            uid=question["uid"],
            answer_type=question["answer_type"],
            gold_scale=question["gold_scale"],
            facts=tuple(FactMatch(**{**fact, "outcome": Outcome(fact["outcome"])})
                        for fact in question["facts"]),
            scale_verdict=ScaleVerdict(question["scale_verdict"]),
        )
        for question in payload.get("questions", ())
    )
    return DocumentResult(
        document_uid=payload["document_uid"],
        source=payload.get("source", ""),
        ok=payload.get("ok", False),
        elapsed_s=payload.get("elapsed_s", 0.0),
        from_cache=payload.get("from_cache", False),
        structure=structure,
        questions=questions,
        error=payload.get("error"),
    )


def write_results(path: Path, meta: RunMeta, results: Sequence[DocumentResult]) -> None:
    """Write a complete results file: one meta line, then one line per document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"record": "run", **asdict(meta)}) + "\n")
        for result in results:
            handle.write(json.dumps(_encode(result)) + "\n")


def read_results(path: Path) -> tuple[RunMeta | None, list[DocumentResult]]:
    """Read back a results file, tolerating a run cut short mid-line."""
    meta: RunMeta | None = None
    results: list[DocumentResult] = []
    if not path.exists():
        return None, results

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                # The last line of an interrupted run. Everything before it is
                # still good, which is the entire point of writing JSONL.
                break
            kind = payload.pop("record", None)
            if kind == "run":
                meta = RunMeta(**payload)
            elif kind == "document":
                results.append(_decode(payload))
    return meta, results


def incompatible(previous: RunMeta, current: RunMeta) -> str:
    """Describe the first setting that makes two runs unmergeable, or "".

    Only the settings that change what is measured are compared. The backend's
    version and model identifiers are deliberately included: TableFormer's
    accurate and fast modes recover different grids, so resuming one run into
    the other would produce a score for a pipeline that never existed.
    """
    for label, before, after in (
        ("backend", previous.backend, current.backend),
        ("backend version", previous.backend_version, current.backend_version),
        ("models", previous.model_versions, current.model_versions),
        ("split", previous.split, current.split),
        ("seed", previous.seed, current.seed),
        ("limit", previous.limit, current.limit),
        ("force_ocr", previous.force_ocr, current.force_ocr),
        ("schema version", previous.schema_version, current.schema_version),
    ):
        if before != after:
            return f"{label} was {before!r}, now {after!r}"
    return ""


def run(
    documents: Sequence[GoldDocument],
    *,
    backend_name: str,
    settings: Settings | None = None,
    out: Path | None = None,
    meta: RunMeta | None = None,
    force_ocr: bool = False,
    use_cache: bool = True,
    resume: bool = False,
    on_result: Callable[[int, int, DocumentResult], None] | None = None,
) -> Iterator[DocumentResult]:
    """Evaluate every document, streaming results and appending them to ``out``.

    Yields as it goes so a caller can report progress on a run that takes
    hours. When ``resume`` is set, documents already present in ``out`` are
    replayed from the file instead of being parsed again.
    """
    settings = settings or Settings()
    backend = registry.get_backend(backend_name)
    backend.warm()

    cache = (
        DocumentCache(settings.cache_dir, enabled=True)
        if use_cache and settings.cache_enabled
        else None
    )

    done: dict[str, DocumentResult] = {}
    if resume and out is not None:
        previous_meta, previous = read_results(out)
        if meta is not None and previous_meta is not None:
            conflict = incompatible(previous_meta, meta)
            if conflict:
                # Appending results from a different configuration would
                # produce one file that averages two backends and says so
                # nowhere. Refusing is the same stance the backend registry
                # takes on an unknown name.
                raise ValueError(
                    f"{out} was produced by a different run ({conflict}). "
                    "Use a different --out, or drop --resume to overwrite."
                )
        done = {result.document_uid: result for result in previous}

    handle = None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fresh = not (resume and out.exists())
        handle = out.open("w" if fresh else "a", encoding="utf-8")
        if fresh and meta is not None:
            handle.write(json.dumps({"record": "run", **asdict(meta)}) + "\n")
            handle.flush()

    try:
        total = len(documents)
        for position, gold in enumerate(documents, start=1):
            previous_result = done.get(gold.uid)
            if previous_result is not None:
                if on_result is not None:
                    on_result(position, total, previous_result)
                yield previous_result
                continue

            result = evaluate_document(
                gold, backend, settings, cache=cache, force_ocr=force_ocr
            )
            if handle is not None:
                handle.write(json.dumps(_encode(result)) + "\n")
                handle.flush()  # a run killed at any point keeps what it had
            if on_result is not None:
                on_result(position, total, result)
            yield result
    finally:
        if handle is not None:
            handle.close()


def backend_meta(backend_name: str) -> tuple[str | None, dict[str, str]]:
    """Version and model identifiers for a backend, for run provenance."""
    try:
        backend = registry.get_backend(backend_name)
        return backend.version, dict(backend.model_versions)
    except Exception:  # provenance must never be the thing that fails a run
        return None, {}


def format_exception() -> str:
    return traceback.format_exc()


__all__ = [
    "SCHEMA_VERSION",
    "DocumentResult",
    "DocumentStructure",
    "QuestionResult",
    "RunMeta",
    "backend_meta",
    "describe_structure",
    "incompatible",
    "evaluate_document",
    "parse_document",
    "read_results",
    "run",
    "score_question",
    "write_results",
]
