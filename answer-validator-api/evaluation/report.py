"""Turning per-document results into the numbers a decision is made from.

The headline is deliberately not a single score. Two numbers carry different
information and averaging them would hide both:

``fact recall``
    Of every string a question needs, how many did the parser recover. This
    moves smoothly and is the right signal when tuning.

``question support``
    Of every question, how many had *all* their facts recovered. This is the
    ceiling on end-to-end accuracy, and it is the number that belongs on a
    slide, because a question missing one operand is simply unanswerable.

Everything else exists to explain a difference between two of those: the
split by answer type, the table-versus-prose breakdown, the scale verdicts,
and the structural counts.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from evaluation.dataset import ANSWER_TYPES
from evaluation.harness import DocumentResult, RunMeta
from evaluation.matching import Outcome, ScaleVerdict


# Gold scales that assert a magnitude, as opposed to the empty string, which
# asserts that the figures are printed at face value.
_DECLARED_SCALES = frozenset({"thousand", "million", "billion"})


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


@dataclass
class Slice:
    """Fact and question tallies for one subset of the questions."""

    name: str
    questions: int = 0
    supported: int = 0
    facts: int = 0
    found: int = 0
    unique: int = 0

    @property
    def fact_recall(self) -> float:
        return _pct(self.found, self.facts)

    @property
    def question_support(self) -> float:
        return _pct(self.supported, self.questions)

    @property
    def unique_rate(self) -> float:
        """Share of recovered facts that matched exactly one place.

        A low rate means the headline recall leans on ambiguous matches — a
        bare ``0.4`` hitting any of nine cells holding 0.4 — and should be
        quoted with that caveat.
        """
        return _pct(self.unique, self.found)


@dataclass
class Report:
    """Everything a run says, aggregated."""

    meta: RunMeta | None = None
    documents: int = 0
    documents_ok: int = 0
    documents_failed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    overall: Slice = field(default_factory=lambda: Slice("overall"))
    by_answer_type: dict[str, Slice] = field(default_factory=dict)

    outcomes: Counter = field(default_factory=Counter)
    scale_verdicts: Counter = field(default_factory=Counter)

    scale_declared_total: int = 0
    scale_declared_correct: int = 0

    pages: int = 0
    blocks: int = 0
    furniture: int = 0
    sections: int = 0
    tables: int = 0
    cells: int = 0
    parsed_nums: int = 0
    scaled_tables: int = 0
    extraction_sources: Counter = field(default_factory=Counter)
    warnings: Counter = field(default_factory=Counter)

    parse_seconds: float = 0.0
    parsed_documents: int = 0
    cached_documents: int = 0

    @property
    def seconds_per_document(self) -> float:
        return self.parse_seconds / self.parsed_documents if self.parsed_documents else 0.0

    @property
    def seconds_per_page(self) -> float:
        return self.parse_seconds / self.pages if self.pages else 0.0

    @property
    def parsed_num_rate(self) -> float:
        """Share of recovered table cells whose figure was also parsed."""
        return _pct(self.parsed_nums, self.cells)

    @property
    def table_fact_share(self) -> float:
        """Of recovered facts, the share found in a table cell rather than prose.

        The clearest single signal of table understanding: the same facts are
        in the same documents either way, so a backend that finds the grid
        moves this number without moving fact recall.
        """
        in_table = (
            self.outcomes[Outcome.TABLE_CELL_TEXT.value]
            + self.outcomes[Outcome.TABLE_CELL_VALUE.value]
        )
        return _pct(in_table, self.overall.found)

    @property
    def scale_scorable(self) -> int:
        """Questions whose gold scale could be compared against a real table.

        A backend that finds no tables scores none of these, and reporting
        that as 0% accuracy would read as "gets every scale wrong" when the
        truth is "never got far enough to have an opinion".
        """
        return sum(
            self.scale_verdicts[verdict.value]
            for verdict in ScaleVerdict
            if verdict is not ScaleVerdict.UNSCORABLE
        )

    @property
    def scale_accuracy(self) -> float | None:
        scorable = self.scale_scorable
        if not scorable:
            return None
        return _pct(self.scale_verdicts[ScaleVerdict.CORRECT.value], scorable)

    @property
    def scale_accuracy_declared(self) -> float | None:
        """Accuracy restricted to questions whose gold states a magnitude.

        Roughly half of TAT-DQA's questions carry an empty gold scale, and a
        parser that never detects a scale at all scores every one of those
        correct. Quoting only the combined number would therefore credit a
        backend for a capability it does not have. This is the number that
        actually tests scale detection: gold says thousands, millions, or
        billions, and the question is whether the caption was read.
        """
        if not self.scale_declared_total:
            return None
        return _pct(self.scale_declared_correct, self.scale_declared_total)


def aggregate(results: Iterable[DocumentResult], meta: RunMeta | None = None) -> Report:
    """Fold per-document results into one report."""
    report = Report(meta=meta)
    report.by_answer_type = {name: Slice(name) for name in ANSWER_TYPES}

    for result in results:
        report.documents += 1
        if not result.ok:
            report.documents_failed += 1
            report.failures.append((result.document_uid, result.error or "unknown"))
            continue

        report.documents_ok += 1
        if result.from_cache:
            report.cached_documents += 1
        else:
            report.parsed_documents += 1
            report.parse_seconds += result.elapsed_s

        structure = result.structure
        report.pages += structure.page_count
        report.blocks += structure.blocks
        report.furniture += structure.furniture
        report.sections += structure.sections
        report.tables += structure.tables
        report.cells += structure.cells
        report.parsed_nums += structure.parsed_nums
        report.scaled_tables += structure.scaled_tables
        report.extraction_sources.update(structure.extraction_sources)
        report.warnings.update(structure.warnings)

        for question in result.questions:
            buckets = [report.overall]
            bucket = report.by_answer_type.get(question.answer_type)
            if bucket is None:
                bucket = report.by_answer_type.setdefault(
                    question.answer_type, Slice(question.answer_type)
                )
            buckets.append(bucket)

            report.scale_verdicts[question.scale_verdict.value] += 1
            if (
                question.scale_verdict is not ScaleVerdict.UNSCORABLE
                and question.gold_scale in _DECLARED_SCALES
            ):
                report.scale_declared_total += 1
                if question.scale_verdict is ScaleVerdict.CORRECT:
                    report.scale_declared_correct += 1

            for entry in buckets:
                entry.questions += 1
                entry.supported += 1 if question.supported else 0
                entry.facts += len(question.facts)
                entry.found += question.found_count
                entry.unique += sum(1 for fact in question.facts if fact.unique)

            for fact in question.facts:
                report.outcomes[fact.outcome.value] += 1

    return report


# -- rendering --------------------------------------------------------------


def _bar(value: float, width: int = 24) -> str:
    filled = int(round(width * value / 100.0))
    return "#" * filled + "." * (width - filled)


def render(report: Report) -> str:
    """A plain-text report, the shape read in a terminal after a run."""
    lines: list[str] = []
    meta = report.meta

    lines.append("=" * 72)
    if meta is not None:
        lines.append(
            f"backend   {meta.backend} {meta.backend_version or ''}".rstrip()
        )
        for name, version in sorted(meta.model_versions.items()):
            lines.append(f"  {name:<12}{version}")
        lines.append(
            f"split     {meta.split}   documents {meta.documents}"
            f"   seed {meta.seed}"
            + (f"   limit {meta.limit}" if meta.limit else "")
            + ("   force_ocr" if meta.force_ocr else "")
        )
    lines.append(
        f"parsed    {report.documents_ok} ok, {report.documents_failed} failed"
        f"   ({report.cached_documents} from cache)"
    )
    if report.parsed_documents:
        lines.append(
            f"speed     {report.seconds_per_document:.1f}s/doc"
            f"   {report.seconds_per_page:.1f}s/page"
        )
    lines.append("=" * 72)
    lines.append("")

    overall = report.overall
    lines.append("EXTRACTION QUALITY  (against TAT-DQA gold facts)")
    lines.append(
        f"  fact recall        {overall.fact_recall:5.1f}%  {_bar(overall.fact_recall)}"
        f"  {overall.found}/{overall.facts}"
    )
    lines.append(
        f"  question support   {overall.question_support:5.1f}%  "
        f"{_bar(overall.question_support)}"
        f"  {overall.supported}/{overall.questions}"
    )
    lines.append(
        f"  unambiguous        {overall.unique_rate:5.1f}%  "
        "of recovered facts matched exactly one place"
    )
    lines.append("")

    lines.append("BY ANSWER TYPE")
    lines.append(f"  {'type':<12}{'questions':>10}{'support':>10}{'fact recall':>14}")
    for name in ANSWER_TYPES:
        entry = report.by_answer_type.get(name)
        if entry is None or not entry.questions:
            continue
        lines.append(
            f"  {name:<12}{entry.questions:>10}"
            f"{entry.question_support:>9.1f}%{entry.fact_recall:>13.1f}%"
        )
    lines.append("")

    lines.append("WHERE THE FACTS WERE FOUND")
    total_facts = overall.facts or 1
    for outcome in Outcome:
        count = report.outcomes[outcome.value]
        lines.append(
            f"  {outcome.value:<20}{count:>7}  {_pct(count, total_facts):5.1f}%"
            + ("   <- lost to furniture demotion" if outcome is Outcome.FURNITURE_ONLY and count else "")
        )
    lines.append(
        f"  table share of recovered facts: {report.table_fact_share:.1f}%"
    )
    lines.append("")

    lines.append("TABLE SCALE  (in thousands / millions - a wrong one is a 1000x error)")
    for verdict in ScaleVerdict:
        count = report.scale_verdicts[verdict.value]
        if count:
            lines.append(f"  {verdict.value:<14}{count:>7}")
    if report.scale_accuracy is None:
        lines.append(
            "  accuracy on scorable questions: n/a  "
            "(no question's facts landed in a table)"
        )
    else:
        lines.append(
            f"  all scorable questions      {report.scale_accuracy:5.1f}%"
            f"  ({report.scale_scorable})"
        )
        declared = report.scale_accuracy_declared
        lines.append(
            "  gold declares a magnitude   "
            + (
                "  n/a"
                if declared is None
                else f"{declared:5.1f}%  ({report.scale_declared_total})"
                "   <- the number that tests caption reading"
            )
        )
    lines.append("")

    lines.append("STRUCTURE FOUND")
    lines.append(f"  pages            {report.pages}")
    lines.append(f"  blocks           {report.blocks}  (+{report.furniture} furniture)")
    lines.append(f"  sections         {report.sections}")
    lines.append(f"  tables           {report.tables}   ({report.scaled_tables} with a declared scale)")
    lines.append(
        f"  cells            {report.cells}   "
        f"{report.parsed_nums} parsed as numbers ({report.parsed_num_rate:.1f}%)"
    )
    if report.extraction_sources:
        lines.append(f"  extraction       {dict(report.extraction_sources)}")
    if report.warnings:
        lines.append(f"  warnings         {dict(report.warnings)}")

    if report.failures:
        lines.append("")
        lines.append(f"FAILURES  ({len(report.failures)})")
        for uid, error in report.failures[:10]:
            lines.append(f"  {uid[:16]}  {error[:80]}")

    return "\n".join(lines)


def compare(reports: Sequence[Report]) -> str:
    """Two or more runs side by side. This is the backend-choice artefact.

    The brief asks for pipeline variants compared with measured results
    rather than intuition; this renders exactly that table.
    """
    if not reports:
        return "nothing to compare"

    def label(report: Report) -> str:
        return report.meta.backend if report.meta else "?"

    names = [label(report) for report in reports]
    width = max(12, max(len(name) for name in names) + 2)

    def row(title: str, values: Sequence[str]) -> str:
        cells = "".join(value.rjust(width) for value in values)
        return f"  {title:<28}{cells}"

    lines = ["BACKEND COMPARISON", ""]
    lines.append(row("", names))
    lines.append("  " + "-" * (28 + width * len(names)))
    lines.append(
        row("documents scored", [str(r.documents_ok) for r in reports])
    )
    lines.append(
        row("fact recall", [f"{r.overall.fact_recall:.1f}%" for r in reports])
    )
    lines.append(
        row("question support", [f"{r.overall.question_support:.1f}%" for r in reports])
    )
    lines.append(
        row("facts found in tables", [f"{r.table_fact_share:.1f}%" for r in reports])
    )
    lines.append(
        row(
            "scale accuracy",
            [
                "n/a" if r.scale_accuracy is None else f"{r.scale_accuracy:.1f}%"
                for r in reports
            ],
        )
    )
    lines.append(
        row(
            "  when gold declares one",
            [
                "n/a"
                if r.scale_accuracy_declared is None
                else f"{r.scale_accuracy_declared:.1f}%"
                for r in reports
            ],
        )
    )
    lines.append("")
    for name in ANSWER_TYPES:
        values = []
        for report in reports:
            entry = report.by_answer_type.get(name)
            values.append(f"{entry.question_support:.1f}%" if entry and entry.questions else "-")
        lines.append(row(f"support: {name}", values))
    lines.append("")
    lines.append(row("tables found", [str(r.tables) for r in reports]))
    lines.append(row("cells found", [str(r.cells) for r in reports]))
    lines.append(row("cells parsed as numbers", [str(r.parsed_nums) for r in reports]))
    lines.append(row("sections found", [str(r.sections) for r in reports]))
    lines.append(
        row("seconds per document", [f"{r.seconds_per_document:.1f}" for r in reports])
    )
    lines.append(row("failed documents", [str(r.documents_failed) for r in reports]))
    return "\n".join(lines)


# -- failure analysis -------------------------------------------------------

# Each cause is decided from the record alone: what the parser found in the
# document, and where the fact was not. Nothing here guesses at intent, so a
# bucket is evidence to look at rather than a verdict.
CAUSES: dict[str, str] = {
    "page_extraction_failed": "the page could not be read at all",
    "no_tables_recovered": "a figure was needed and the parser found no table on the page",
    "table_found_cell_missed": "a table was recovered but this cell was not, or was misread",
    "demoted_to_furniture": "present, but in a block excluded from reading order",
    "prose_not_matched": "a phrase the extracted text does not contain",
}


def diagnose(results: Sequence[DocumentResult]) -> Counter:
    """Bucket every unrecovered fact by the cause the record supports."""
    causes: Counter = Counter()
    for result in results:
        if not result.ok:
            causes["document_failed"] += 1
            continue
        has_tables = result.structure.tables > 0
        page_failed = result.structure.extraction_sources.get("failed", 0) > 0
        for question in result.questions:
            for fact in question.facts:
                if fact.found:
                    continue
                if fact.outcome is Outcome.FURNITURE_ONLY:
                    causes["demoted_to_furniture"] += 1
                elif page_failed:
                    causes["page_extraction_failed"] += 1
                elif fact.numeric and not has_tables:
                    causes["no_tables_recovered"] += 1
                elif fact.numeric:
                    causes["table_found_cell_missed"] += 1
                else:
                    causes["prose_not_matched"] += 1
    return causes


def worst_documents(
    results: Sequence[DocumentResult], limit: int = 5
) -> list[DocumentResult]:
    """The documents whose questions are least supported, worst first.

    Ranked by the count of unsupported questions rather than by rate, so a
    document that loses six questions outranks one that loses its only one.
    """
    scored = [
        (sum(1 for q in r.questions if not q.supported), r)
        for r in results
        if r.questions or not r.ok
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [result for count, result in scored[:limit] if count or not result.ok]


def render_failures(results: Sequence[DocumentResult], limit: int = 5) -> str:
    """Why the misses happened, and which documents to open first.

    The brief asks for failures root-caused to a specific pipeline stage. This
    does that for the extraction stage: every lost fact is attributed to
    something observable in the parse, and the worst documents are named so
    the next step is opening one, not guessing.
    """
    lines = ["FAILURE ANALYSIS", ""]

    causes = diagnose(results)
    total = sum(causes.values())
    if not total:
        return "FAILURE ANALYSIS\n\n  no unrecovered facts"

    lines.append(f"  {total} unrecovered facts, by cause:")
    for name, count in causes.most_common():
        explanation = CAUSES.get(name, "")
        lines.append(
            f"    {name:<26}{count:>6}  {_pct(count, total):5.1f}%  {explanation}"
        )
    lines.append("")
    lines.append(
        "  Note: prose_not_matched includes wording the extractor rendered"
    )
    lines.append(
        "  differently (ligatures, hyphenation, spacing), not only lost text."
    )
    lines.append("")

    lines.append("  Worst documents (open these first):")
    for result in worst_documents(results, limit):
        if not result.ok:
            lines.append(f"    {result.document_uid[:16]}  FAILED  {result.error}")
            continue
        unsupported = [q for q in result.questions if not q.supported]
        lines.append(
            f"    {result.document_uid[:16]}  {result.source[:34]:<34}"
            f"  {len(unsupported)}/{len(result.questions)} questions unsupported"
            f"  tables={result.structure.tables}"
        )
        missing = [
            fact.fact
            for question in unsupported
            for fact in question.facts
            if not fact.found
        ]
        for value in missing[:4]:
            lines.append(f"        missing: {value[:64]!r}")
    return "\n".join(lines)


def common_documents(runs: Sequence[Sequence[DocumentResult]]) -> set[str]:
    """Document ids every run scored."""
    if not runs:
        return set()
    shared = {result.document_uid for result in runs[0]}
    for run in runs[1:]:
        shared &= {result.document_uid for result in run}
    return shared


def restrict(results: Sequence[DocumentResult], keep: set[str]) -> list[DocumentResult]:
    """Drop results for documents outside ``keep``, preserving order."""
    return [result for result in results if result.document_uid in keep]


def coverage_warning(runs: Sequence[Sequence[DocumentResult]]) -> str:
    """Warn when the runs being compared do not cover the same documents.

    Comparing a 274-document run against a 10-document one produces a tidy
    table in which every difference is confounded by which documents each
    backend happened to see. Silence there would be the worst outcome the
    harness could produce, so the mismatch is stated and the fix named.
    """
    if len(runs) < 2:
        return ""
    sizes = [len({result.document_uid for result in run}) for run in runs]
    shared = len(common_documents(runs))
    if all(size == shared for size in sizes):
        return ""
    return (
        f"WARNING: these runs cover different documents ({sizes} scored, "
        f"{shared} in common). Differences below are confounded by the sample. "
        "Re-run with --common to compare only the shared documents."
    )


def to_dict(report: Report) -> dict:
    """The report as JSON, for a dashboard or a diff between runs."""
    return {
        "meta": report.meta.__dict__ if report.meta else None,
        "documents": {
            "total": report.documents,
            "ok": report.documents_ok,
            "failed": report.documents_failed,
            "from_cache": report.cached_documents,
        },
        "quality": {
            "fact_recall": round(report.overall.fact_recall, 2),
            "question_support": round(report.overall.question_support, 2),
            "unique_match_rate": round(report.overall.unique_rate, 2),
            "table_fact_share": round(report.table_fact_share, 2),
            "scale_accuracy": (
                None if report.scale_accuracy is None else round(report.scale_accuracy, 2)
            ),
            "scale_scorable": report.scale_scorable,
            "scale_accuracy_declared": (
                None
                if report.scale_accuracy_declared is None
                else round(report.scale_accuracy_declared, 2)
            ),
            "scale_declared_scorable": report.scale_declared_total,
            "facts": report.overall.facts,
            "facts_found": report.overall.found,
            "questions": report.overall.questions,
            "questions_supported": report.overall.supported,
        },
        "by_answer_type": {
            name: {
                "questions": entry.questions,
                "question_support": round(entry.question_support, 2),
                "fact_recall": round(entry.fact_recall, 2),
            }
            for name, entry in report.by_answer_type.items()
            if entry.questions
        },
        "outcomes": dict(report.outcomes),
        "scale_verdicts": dict(report.scale_verdicts),
        "structure": {
            "pages": report.pages,
            "blocks": report.blocks,
            "furniture": report.furniture,
            "sections": report.sections,
            "tables": report.tables,
            "cells": report.cells,
            "parsed_nums": report.parsed_nums,
            "scaled_tables": report.scaled_tables,
            "extraction_sources": dict(report.extraction_sources),
            "warnings": dict(report.warnings),
        },
        "performance": {
            "seconds_per_document": round(report.seconds_per_document, 3),
            "seconds_per_page": round(report.seconds_per_page, 3),
        },
        "failures": [{"document_uid": uid, "error": error} for uid, error in report.failures],
    }


def to_json(report: Report) -> str:
    return json.dumps(to_dict(report), indent=2)


__all__ = [
    "CAUSES",
    "Report",
    "Slice",
    "aggregate",
    "common_documents",
    "compare",
    "coverage_warning",
    "diagnose",
    "restrict",
    "render",
    "render_failures",
    "to_dict",
    "to_json",
    "worst_documents",
]
