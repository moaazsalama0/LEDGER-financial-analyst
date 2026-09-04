"""Tests for the rule that decides whether a gold fact was recovered.

This is the half of the harness most likely to be wrong, and it costs
milliseconds to check. Every quality number the project reports is this rule
applied 10,000 times, so a matcher that is quietly too generous produces a
backend comparison that looks rigorous and means nothing.
"""

from __future__ import annotations

import pytest

from evaluation.matching import (
    Outcome,
    ScaleVerdict,
    build_index,
    match_fact,
    normalize,
    score_scale,
)
from ledger_doc_contract.v1.enums import BlockType, ExtractionSource
from ledger_doc_contract.v1.models import (
    Block,
    Cell,
    NumericValue,
    Page,
    ProcessedDocument,
    ProcessingInfo,
    Table,
    TableUnits,
)


def make_document(
    *,
    blocks: list[Block] | None = None,
    reading_order: list[str] | None = None,
    tables: list[Table] | None = None,
) -> ProcessedDocument:
    """A minimal document, built from the contract rather than from a PDF."""
    blocks = blocks or []
    order = reading_order if reading_order is not None else [b.block_id for b in blocks]
    return ProcessedDocument(
        document_id="doc_test",
        filename="test.pdf",
        content_sha256="0" * 64,
        page_count=1,
        pages=(
            Page(
                page_number=1,
                width=612.0,
                height=792.0,
                extraction_source=ExtractionSource.TEXT_LAYER,
                blocks=tuple(blocks),
                reading_order=tuple(order),
            ),
        ),
        tables=tuple(tables or []),
        processing=ProcessingInfo(
            backend="fake",
            service_version="0.1.0",
            processed_at="2026-01-01T00:00:00Z",
            duration_ms=0,
        ),
    )


def text_block(block_id: str, text: str, index: int = 0, *, kind: BlockType = BlockType.TEXT) -> Block:
    return Block(
        block_id=block_id, page_number=1, type=kind, text=text, order_index=index
    )


def numeric_table(
    cells: list[tuple[int, int, str, float | None]],
    *,
    table_id: str = "t000",
    scale: float = 1.0,
) -> Table:
    return Table(
        table_id=table_id,
        block_id="p0001_b000",
        page_number=1,
        n_rows=max((row for row, _, _, _ in cells), default=-1) + 1,
        n_cols=max((col for _, col, _, _ in cells), default=-1) + 1,
        units=TableUnits(scale=scale),
        cells=tuple(
            Cell(
                cell_id=f"{table_id}_r{row}_c{col}",
                row=row,
                col=col,
                text=text,
                value=None if num is None else NumericValue(num=num, scaled=num * scale),
            )
            for row, col, text, num in cells
        ),
    )


class TestNormalize:
    def test_collapses_whitespace_and_case(self) -> None:
        assert normalize("  Total   Revenue\n") == "total revenue"

    def test_folds_dash_variants_onto_one(self) -> None:
        # Reports and the gold file disagree about which dash to use for the
        # same printed glyph; a match must not turn on that choice.
        assert normalize("year–end") == normalize("year-end")
        assert normalize("—") == "-"

    def test_folds_unicode_presentation_forms(self) -> None:
        assert normalize("１２３") == "123"


class TestTableCellMatching:
    def test_exact_cell_text_is_the_strongest_result(self) -> None:
        document = make_document(tables=[numeric_table([(0, 0, "1,234", 1234.0)])])
        match = match_fact("1,234", build_index(document))
        assert match.outcome is Outcome.TABLE_CELL_TEXT
        assert match.cell_id == "t000_r0_c0"
        assert match.found and match.unique

    def test_same_number_written_differently_still_matches(self) -> None:
        # The finance convention: the page prints (1,234), the gold says -1,234.
        document = make_document(tables=[numeric_table([(0, 0, "(1,234)", -1234.0)])])
        match = match_fact("-1,234", build_index(document))
        assert match.outcome is Outcome.TABLE_CELL_VALUE
        assert match.numeric

    def test_currency_symbol_does_not_defeat_a_cell_match(self) -> None:
        document = make_document(tables=[numeric_table([(0, 0, "1,234", 1234.0)])])
        assert match_fact("$1,234", build_index(document)).found

    def test_sign_is_not_ignored(self) -> None:
        # A parser that loses the parentheses must be scored as wrong, not
        # forgiven — that is the exact error this project exists to prevent.
        document = make_document(tables=[numeric_table([(0, 0, "1,234", 1234.0)])])
        assert match_fact("(1,234)", build_index(document)).outcome is Outcome.NOT_FOUND

    def test_repeated_value_is_reported_as_ambiguous(self) -> None:
        document = make_document(
            tables=[numeric_table([(0, 0, "5", 5.0), (0, 1, "5", 5.0)])]
        )
        match = match_fact("5", build_index(document))
        assert match.found
        assert match.candidates == 2
        assert not match.unique


class TestBlockTextMatching:
    def test_prose_fact_is_found_in_a_block(self) -> None:
        document = make_document(
            blocks=[text_block("p0001_b000", "Software is carried at cost less amortisation.")]
        )
        match = match_fact("at cost less amortisation.", build_index(document))
        assert match.outcome is Outcome.BLOCK_TEXT
        assert match.block_id == "p0001_b000"

    def test_a_number_is_not_matched_inside_a_longer_number(self) -> None:
        # Without the word-edge guard, 181 matches 1,181,204 and recall is a
        # fiction.
        document = make_document(blocks=[text_block("p0001_b000", "Total was 1,181,204 for the year")])
        assert match_fact("181", build_index(document)).outcome is Outcome.NOT_FOUND

    def test_a_word_is_not_matched_inside_a_longer_word(self) -> None:
        document = make_document(blocks=[text_block("p0001_b000", "the network segment")])
        assert match_fact("netw", build_index(document)).outcome is Outcome.NOT_FOUND

    def test_detached_currency_symbol_still_counts_as_recovered(self) -> None:
        # A text layer that emits "$ 278,057" has recovered the figure. Scoring
        # it as a miss would report a spacing artefact as a lost fact.
        document = make_document(
            blocks=[text_block("p0001_b000", "costs amounted to $ 278,057 for the year")]
        )
        match = match_fact("$278,057", build_index(document))
        assert match.outcome is Outcome.BLOCK_TEXT_VALUE
        assert match.found

    def test_a_cell_beats_prose_when_both_hold_the_fact(self) -> None:
        document = make_document(
            blocks=[text_block("p0001_b000", "revenue of 1,234 this year")],
            tables=[numeric_table([(0, 0, "1,234", 1234.0)])],
        )
        assert match_fact("1,234", build_index(document)).outcome is Outcome.TABLE_CELL_TEXT


class TestFurniture:
    def test_a_fact_only_in_furniture_counts_as_lost(self) -> None:
        # Page furniture is excluded from reading_order, so it never reaches a
        # retrieval chunk. Counting it as found would hide the bug.
        header = text_block("p0001_b000", "Annual Report 2019 page 12", kind=BlockType.PAGE_HEADER)
        body = text_block("p0001_b001", "Nothing relevant here.", 1)
        document = make_document(blocks=[header, body], reading_order=["p0001_b001"])
        match = match_fact("Annual Report 2019", build_index(document))
        assert match.outcome is Outcome.FURNITURE_ONLY
        assert not match.found


class TestMissing:
    def test_absent_fact_is_not_found(self) -> None:
        document = make_document(blocks=[text_block("p0001_b000", "unrelated prose")])
        assert match_fact("9,999", build_index(document)).outcome is Outcome.NOT_FOUND

    def test_empty_fact_is_not_found(self) -> None:
        document = make_document(blocks=[text_block("p0001_b000", "text")])
        assert match_fact("", build_index(document)).outcome is Outcome.NOT_FOUND


class TestScaleScoring:
    @pytest.mark.parametrize(
        ("gold", "factor", "scales", "expected"),
        [
            ("million", 1e6, (1e6,), ScaleVerdict.CORRECT),
            ("thousand", 1e3, (1.0,), ScaleVerdict.MISSED),
            ("thousand", 1e3, (1e6,), ScaleVerdict.WRONG),
            ("", 1.0, (1e6,), ScaleVerdict.SPURIOUS),
            ("", 1.0, (1.0,), ScaleVerdict.CORRECT),
            ("percent", None, (1e6,), ScaleVerdict.UNSCORABLE),
            ("million", 1e6, (), ScaleVerdict.UNSCORABLE),
        ],
    )
    def test_verdicts(
        self, gold: str, factor: float | None, scales: tuple, expected: ScaleVerdict
    ) -> None:
        assert score_scale(gold, factor, scales) is expected

    def test_any_matching_table_is_enough(self) -> None:
        # Facts can legitimately span tables scaled differently; agreement by
        # the one that holds the answer's figures is what matters.
        assert score_scale("million", 1e6, (1.0, 1e6)) is ScaleVerdict.CORRECT
