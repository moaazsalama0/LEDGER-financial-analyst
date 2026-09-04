"""Tests for aggregation, the results file, and the rendered report.

Two properties matter beyond arithmetic. A results file must survive being
killed mid-run, because a corpus run is measured in hours. And a report must
not turn "never got far enough to have an opinion" into a confident 0%.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.harness import (
    DocumentResult,
    DocumentStructure,
    QuestionResult,
    RunMeta,
    incompatible,
    read_results,
    write_results,
)
from evaluation.matching import FactMatch, Outcome, ScaleVerdict
from evaluation.report import (
    aggregate,
    compare,
    diagnose,
    render,
    render_failures,
    to_dict,
    worst_documents,
)


def fact(outcome: Outcome, *, candidates: int = 1, text: str = "1,234") -> FactMatch:
    return FactMatch(fact=text, outcome=outcome, numeric=True, candidates=candidates)


def question(
    *facts: FactMatch,
    answer_type: str = "arithmetic",
    scale: str = "million",
    verdict: ScaleVerdict = ScaleVerdict.CORRECT,
) -> QuestionResult:
    return QuestionResult(
        uid="q",
        answer_type=answer_type,
        gold_scale=scale,
        facts=facts,
        scale_verdict=verdict,
    )


def document(*questions: QuestionResult, ok: bool = True, **structure) -> DocumentResult:
    return DocumentResult(
        document_uid="doc",
        source="report.pdf",
        ok=ok,
        elapsed_s=1.0,
        structure=DocumentStructure(**structure),
        questions=questions,
        error=None if ok else "BackendFailure: boom",
    )


class TestQuestionSupport:
    def test_a_question_needs_every_fact(self) -> None:
        # An arithmetic question missing one operand is unanswerable, not
        # half-answerable.
        partial = question(fact(Outcome.TABLE_CELL_TEXT), fact(Outcome.NOT_FOUND))
        assert not partial.supported
        assert partial.found_count == 1

    def test_all_facts_found_is_supported(self) -> None:
        assert question(
            fact(Outcome.TABLE_CELL_TEXT), fact(Outcome.BLOCK_TEXT_VALUE)
        ).supported

    def test_a_question_with_no_facts_is_not_supported(self) -> None:
        assert not question().supported

    def test_furniture_does_not_count_as_found(self) -> None:
        assert not question(fact(Outcome.FURNITURE_ONLY)).supported


class TestAggregate:
    def test_counts_facts_and_questions(self) -> None:
        report = aggregate(
            [
                document(
                    question(fact(Outcome.TABLE_CELL_TEXT)),
                    question(fact(Outcome.NOT_FOUND), fact(Outcome.BLOCK_TEXT)),
                )
            ]
        )
        assert report.overall.questions == 2
        assert report.overall.supported == 1
        assert report.overall.facts == 3
        assert report.overall.found == 2
        assert report.overall.question_support == 50.0
        assert report.overall.fact_recall == pytest.approx(2 / 3 * 100)

    def test_splits_by_answer_type(self) -> None:
        report = aggregate(
            [
                document(
                    question(fact(Outcome.TABLE_CELL_TEXT), answer_type="arithmetic"),
                    question(fact(Outcome.NOT_FOUND), answer_type="span"),
                )
            ]
        )
        assert report.by_answer_type["arithmetic"].question_support == 100.0
        assert report.by_answer_type["span"].question_support == 0.0

    def test_table_share_measures_table_understanding(self) -> None:
        report = aggregate(
            [
                document(
                    question(fact(Outcome.TABLE_CELL_TEXT)),
                    question(fact(Outcome.BLOCK_TEXT_VALUE)),
                )
            ]
        )
        assert report.table_fact_share == 50.0

    def test_ambiguous_matches_are_tracked_separately(self) -> None:
        report = aggregate(
            [
                document(
                    question(fact(Outcome.TABLE_CELL_VALUE, candidates=9)),
                    question(fact(Outcome.TABLE_CELL_TEXT, candidates=1)),
                )
            ]
        )
        assert report.overall.found == 2
        assert report.overall.unique == 1
        assert report.overall.unique_rate == 50.0

    def test_a_failed_document_is_counted_not_dropped(self) -> None:
        report = aggregate([document(ok=False), document(question(fact(Outcome.BLOCK_TEXT)))])
        assert report.documents == 2
        assert report.documents_ok == 1
        assert report.documents_failed == 1
        assert report.failures[0][1].startswith("BackendFailure")

    def test_structure_totals_add_up(self) -> None:
        report = aggregate(
            [
                document(tables=2, cells=20, parsed_nums=12, page_count=1),
                document(tables=3, cells=30, parsed_nums=18, page_count=2),
            ]
        )
        assert (report.tables, report.cells, report.parsed_nums) == (5, 50, 30)
        assert report.parsed_num_rate == 60.0
        assert report.pages == 3


class TestScaleReporting:
    def test_accuracy_over_scorable_questions_only(self) -> None:
        report = aggregate(
            [
                document(
                    question(fact(Outcome.TABLE_CELL_TEXT), verdict=ScaleVerdict.CORRECT),
                    question(fact(Outcome.TABLE_CELL_TEXT), verdict=ScaleVerdict.MISSED),
                    question(fact(Outcome.NOT_FOUND), verdict=ScaleVerdict.UNSCORABLE),
                )
            ]
        )
        assert report.scale_scorable == 2
        assert report.scale_accuracy == 50.0

    def test_declared_scales_are_scored_separately(self) -> None:
        # Half of TAT-DQA carries an empty gold scale, and a parser that never
        # detects one scores every one of those correct. The headline would
        # credit a capability the backend does not have.
        report = aggregate(
            [
                document(
                    question(
                        fact(Outcome.TABLE_CELL_TEXT),
                        scale="",
                        verdict=ScaleVerdict.CORRECT,
                    ),
                    question(
                        fact(Outcome.TABLE_CELL_TEXT),
                        scale="million",
                        verdict=ScaleVerdict.MISSED,
                    ),
                )
            ]
        )
        assert report.scale_accuracy == 50.0
        assert report.scale_declared_total == 1
        assert report.scale_accuracy_declared == 0.0

    def test_percent_questions_never_reach_the_declared_bucket(self) -> None:
        report = aggregate(
            [
                document(
                    question(
                        fact(Outcome.TABLE_CELL_TEXT),
                        scale="percent",
                        verdict=ScaleVerdict.UNSCORABLE,
                    )
                )
            ]
        )
        assert report.scale_declared_total == 0
        assert report.scale_accuracy_declared is None

    def test_nothing_scorable_reports_none_not_zero(self) -> None:
        # A backend that finds no tables has no opinion on scale. Rendering
        # that as 0% would read as "gets every scale wrong".
        report = aggregate(
            [document(question(fact(Outcome.NOT_FOUND), verdict=ScaleVerdict.UNSCORABLE))]
        )
        assert report.scale_accuracy is None
        assert "n/a" in render(report)


class TestResultsFile:
    def test_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "run.jsonl"
        meta = RunMeta(
            backend="docling",
            backend_version="2.0",
            model_versions={"layout": "docLayNet"},
            split="dev",
            seed=7,
            limit=10,
            documents=1,
            force_ocr=False,
        )
        results = [document(question(fact(Outcome.TABLE_CELL_TEXT)), tables=1, cells=4)]
        write_results(path, meta, results)

        loaded_meta, loaded = read_results(path)
        assert loaded_meta == meta
        assert len(loaded) == 1
        assert loaded[0].questions[0].facts[0].outcome is Outcome.TABLE_CELL_TEXT
        assert loaded[0].structure.cells == 4
        assert aggregate(loaded, loaded_meta).overall.supported == 1

    def test_a_truncated_final_line_keeps_everything_before_it(
        self, tmp_path: Path
    ) -> None:
        # The point of JSONL: a run killed at document 200 of 274 has 200
        # usable results, not none.
        path = tmp_path / "run.jsonl"
        meta = RunMeta(
            backend="textonly", backend_version=None, model_versions={},
            split="dev", seed=0, limit=None, documents=2, force_ocr=False,
        )
        write_results(path, meta, [document(question(fact(Outcome.BLOCK_TEXT)))])
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"record": "document", "document_uid": "trunc')

        loaded_meta, loaded = read_results(path)
        assert loaded_meta is not None
        assert len(loaded) == 1

    def test_absent_file_reads_as_empty(self, tmp_path: Path) -> None:
        assert read_results(tmp_path / "nope.jsonl") == (None, [])


class TestResumeCompatibility:
    def _meta(self, **overrides) -> RunMeta:
        base = dict(
            backend="docling",
            backend_version="2.0",
            model_versions={"table": "tableformer-accurate"},
            split="dev",
            seed=0,
            limit=None,
            documents=274,
            force_ocr=False,
        )
        base.update(overrides)
        return RunMeta(**base)

    def test_identical_configuration_is_compatible(self) -> None:
        assert incompatible(self._meta(), self._meta()) == ""

    def test_document_count_alone_does_not_block_a_resume(self) -> None:
        # A resumed run legitimately reports a different remaining count.
        assert incompatible(self._meta(), self._meta(documents=100)) == ""

    def test_a_different_backend_is_refused(self) -> None:
        assert "backend" in incompatible(self._meta(), self._meta(backend="textonly"))

    def test_a_different_table_mode_is_refused(self) -> None:
        # TableFormer's accurate and fast modes recover different grids;
        # resuming one into the other scores a pipeline that never ran.
        conflict = incompatible(
            self._meta(), self._meta(model_versions={"table": "tableformer-fast"})
        )
        assert "models" in conflict

    def test_a_different_sample_is_refused(self) -> None:
        assert "seed" in incompatible(self._meta(), self._meta(seed=9))
        assert "limit" in incompatible(self._meta(), self._meta(limit=50))


class TestFailureAnalysis:
    def test_a_figure_lost_where_no_table_was_found_blames_the_layout_model(
        self,
    ) -> None:
        results = [document(question(fact(Outcome.NOT_FOUND)), tables=0)]
        assert diagnose(results)["no_tables_recovered"] == 1

    def test_a_figure_lost_despite_a_table_blames_the_cell(self) -> None:
        results = [document(question(fact(Outcome.NOT_FOUND)), tables=2)]
        assert diagnose(results)["table_found_cell_missed"] == 1

    def test_furniture_is_its_own_cause(self) -> None:
        results = [document(question(fact(Outcome.FURNITURE_ONLY)), tables=1)]
        assert diagnose(results)["demoted_to_furniture"] == 1

    def test_a_failed_page_outranks_the_other_explanations(self) -> None:
        # There is no point blaming the table model for a page nothing could
        # read.
        results = [
            document(
                question(fact(Outcome.NOT_FOUND)),
                tables=0,
                extraction_sources={"failed": 1},
            )
        ]
        assert diagnose(results)["page_extraction_failed"] == 1

    def test_a_prose_miss_is_not_blamed_on_tables(self) -> None:
        prose = FactMatch(fact="a long phrase", outcome=Outcome.NOT_FOUND, numeric=False)
        results = [document(question(prose), tables=0)]
        assert diagnose(results)["prose_not_matched"] == 1

    def test_recovered_facts_are_not_diagnosed(self) -> None:
        results = [document(question(fact(Outcome.TABLE_CELL_TEXT)), tables=1)]
        assert sum(diagnose(results).values()) == 0

    def test_worst_documents_ranks_by_questions_lost(self) -> None:
        few = DocumentResult(
            document_uid="few", source="", ok=True, elapsed_s=0.0,
            questions=(question(fact(Outcome.NOT_FOUND)),),
        )
        many = DocumentResult(
            document_uid="many", source="", ok=True, elapsed_s=0.0,
            questions=tuple(question(fact(Outcome.NOT_FOUND)) for _ in range(4)),
        )
        assert [d.document_uid for d in worst_documents([few, many], 2)] == ["many", "few"]

    def test_fully_supported_documents_are_not_listed(self) -> None:
        good = document(question(fact(Outcome.TABLE_CELL_TEXT)))
        assert worst_documents([good]) == []

    def test_render_is_ascii_and_names_a_document(self) -> None:
        results = [document(question(fact(Outcome.NOT_FOUND)), tables=0)]
        text = render_failures(results)
        assert text.isascii()
        assert "no_tables_recovered" in text
        assert "doc" in text

    def test_nothing_to_report_says_so(self) -> None:
        results = [document(question(fact(Outcome.TABLE_CELL_TEXT)), tables=1)]
        assert "no unrecovered facts" in render_failures(results)


class TestRendering:
    def _report(self, backend: str):
        meta = RunMeta(
            backend=backend, backend_version="1.0", model_versions={},
            split="dev", seed=0, limit=None, documents=1, force_ocr=False,
        )
        return aggregate([document(question(fact(Outcome.TABLE_CELL_TEXT)), tables=1)], meta)

    def test_render_is_ascii_only(self) -> None:
        # The report is read in a Windows console, where a stray em dash
        # renders as a replacement character.
        text = render(self._report("docling"))
        assert text.isascii(), [c for c in text if not c.isascii()][:5]

    def test_render_names_the_headline_numbers(self) -> None:
        text = render(self._report("docling"))
        assert "fact recall" in text
        assert "question support" in text

    def test_compare_puts_backends_side_by_side(self) -> None:
        text = compare([self._report("textonly"), self._report("docling")])
        assert "textonly" in text and "docling" in text
        assert "question support" in text
        assert text.isascii()

    def test_compare_of_nothing_does_not_crash(self) -> None:
        assert compare([]) == "nothing to compare"

    def test_json_report_carries_the_headline_numbers(self) -> None:
        payload = to_dict(self._report("docling"))
        assert payload["quality"]["question_support"] == 100.0
        assert payload["meta"]["backend"] == "docling"
        assert payload["structure"]["tables"] == 1
