"""Tests for magnitude detection — the Phase 2.1 intervention.

Phase 2 measured the caption-only detector at 37.2% on questions whose gold
declares a magnitude. Every failure here is a thousandfold error in an answer
that still carries a correct-looking citation, which makes this the highest
-consequence parsing in the service.

The strings exercised below are taken from the TAT-DQA dev corpus rather than
invented, so this suite is also the record of what the phrase actually looks
like in a filing.
"""

from __future__ import annotations

import pytest

from app.backends.base import RawCell, RawTable
from app.processing.finance import parse_scale, parse_value
from app.processing.scale import ScaleSource, detect_scale, find_candidates
from app.processing.tables import build_table


def build(
    *,
    caption: str | None = None,
    cells: list[RawCell] | None = None,
    preceding: tuple[str, ...] = (),
    header_rows: int = 1,
):
    """A one-table document fragment, built the way ``assemble`` builds it."""
    cells = cells or [
        RawCell(row=0, col=0, text="", is_header=True),
        RawCell(row=0, col=1, text="2019", is_header=True),
        RawCell(row=1, col=0, text="Revenue"),
        RawCell(row=1, col=1, text="1,234"),
    ]
    raw = RawTable(
        n_rows=max(c.row for c in cells) + 1,
        n_cols=max(c.col for c in cells) + 1,
        cells=tuple(cells),
        header_rows=header_rows,
        caption=caption,
    )
    return build_table(
        raw,
        table_id="t000",
        block_id="p0001_b000",
        page_number=1,
        section_id=None,
        bbox=None,
        preceding_texts=preceding,
    )


class TestRequiredCases:
    """The cases named in the Phase 2.1 brief."""

    def test_1_caption_amounts_in_millions(self) -> None:
        detection = detect_scale(caption="Amounts in millions")
        assert detection.units.scale == pytest.approx(1e6)
        assert detection.source is ScaleSource.CAPTION

    def test_2_column_header_dollar_in_thousands(self) -> None:
        detection = detect_scale(headers=["$ in thousands"])
        assert detection.units.scale == pytest.approx(1e3)
        assert detection.source is ScaleSource.HEADER
        assert detection.units.currency == "USD"

    def test_3_currency_and_magnitude_usd_in_millions(self) -> None:
        detection = detect_scale(headers=["USD in millions"])
        assert detection.units.scale == pytest.approx(1e6)
        assert detection.units.currency == "USD"

    def test_4_paragraph_immediately_before_the_table(self) -> None:
        detection = detect_scale(preceding=["The following amounts are in thousands."])
        assert detection.units.scale == pytest.approx(1e3)
        assert detection.source is ScaleSource.PRECEDING_TEXT

    def test_5_no_scale_present(self) -> None:
        detection = detect_scale(
            caption="Components of income tax expense",
            headers=["2019", "2018"],
            row_stems=["Current federal", "Deferred"],
        )
        assert detection.units.scale == pytest.approx(1.0)
        assert detection.units.scale_label is None
        assert detection.source is None

    def test_6_an_unrelated_million_is_not_a_table_scale(self) -> None:
        # A figure in a sentence is not a unit declaration. Reading it as one
        # would multiply an entire table by a million.
        detection = detect_scale(
            preceding=["The Company recorded a $5 million impairment charge."]
        )
        assert detection.units.scale == pytest.approx(1.0)
        assert detection.source is None

    def test_7_conflicting_signals_resolve_by_precedence_and_are_flagged(self) -> None:
        detection = detect_scale(
            headers=["(in millions)"],
            row_stems=["(in thousands)"],
            preceding=["(in billions)"],
        )
        assert detection.units.scale == pytest.approx(1e6), "header outranks the rest"
        assert detection.source is ScaleSource.HEADER
        # Different sources disagreeing is what precedence exists to settle,
        # so this is resolved, not ambiguous...
        assert not detection.ambiguous
        # ...but every candidate is kept so the disagreement stays inspectable.
        assert {c.scale for c in detection.candidates} == {1e3, 1e6, 1e9}

    def test_7b_a_conflict_inside_one_source_is_ambiguous(self) -> None:
        # No rule can settle a table whose own headers disagree, so it is
        # flagged rather than silently decided.
        detection = detect_scale(headers=["(in millions)", "(in thousands)"])
        assert detection.ambiguous
        assert detection.units.scale in (1e3, 1e6)

    def test_7c_the_ambiguous_pick_is_deterministic(self) -> None:
        headers = ["(in millions)", "(in thousands)"]
        first = detect_scale(headers=headers).units.scale
        for _ in range(5):
            assert detect_scale(headers=list(headers)).units.scale == first

    def test_7d_the_majority_declaration_wins_within_a_source(self) -> None:
        detection = detect_scale(
            headers=["(in millions)", "(in millions)", "(in thousands)"]
        )
        assert detection.units.scale == pytest.approx(1e6)
        assert detection.ambiguous, "the minority reading is still reported"

    def test_8_negative_values_keep_their_sign_under_scale(self) -> None:
        table = build(caption="(in millions)")
        value = parse_value("(1,234)", units=table.units)
        assert value is not None
        assert value.num == pytest.approx(-1234.0), "num stays as printed"
        assert value.scaled == pytest.approx(-1_234_000_000.0)

    def test_9_comma_formatted_values_scale(self) -> None:
        table = build(caption="(in thousands)")
        value = parse_value("1,181,204", units=table.units)
        assert value is not None
        assert value.num == pytest.approx(1181204.0)
        assert value.scaled == pytest.approx(1_181_204_000.0)

    def test_10_decimal_values_scale(self) -> None:
        table = build(caption="(in millions)")
        value = parse_value("3,355.1", units=table.units)
        assert value is not None
        assert value.num == pytest.approx(3355.1)
        assert value.scaled == pytest.approx(3_355_100_000.0)

    @pytest.mark.parametrize(
        ("text", "num"),
        [("1,181,204", 1181204.0), ("$278,057", 278057.0), ("(567)", -567.0)],
    )
    def test_11_known_regression_values_are_unchanged(
        self, text: str, num: float
    ) -> None:
        # The printed figure must survive the scale work untouched: `num` is
        # always the number as printed, whatever multiplier the table declares.
        for caption in (None, "(in thousands)", "(in millions)"):
            value = parse_value(text, units=build(caption=caption).units)
            assert value is not None
            assert value.num == pytest.approx(num)


class TestCorpusPhrases:
    """Strings taken verbatim from the TAT-DQA dev split."""

    @pytest.mark.parametrize(
        ("text", "scale"),
        [
            ("(in millions)", 1e6),
            ("(In thousands)", 1e3),
            ("(dollars in thousands)", 1e3),
            ("($ in millions)", 1e6),
            ("(Dollars in millions)", 1e6),
            ("(Amounts in thousands)", 1e3),
            ("Year Ended December 31, 2019 (In millions)", 1e6),
            ("Fiscal 2019 (in millions)", 1e6),
            ("Years Ended December 31, 2019 (dollars in thousands)", 1e3),
            ("(In thousands, except percentages)", 1e3),
            ("(In millions, except percentages and per share amounts)", 1e6),
            ("(In thousands of Canadian dollars)", 1e3),
            ("(in thousands of U.S. dollars)", 1e3),
            ("(in millions of €)", 1e6),
            ("($ in millions) At December 31:", 1e6),
            ("$ million", 1e6),
            ("$ MILLION", 1e6),
            ("RMB'Million", 1e6),
            ("2019 RMB'Million", 1e6),
            ("€ million", 1e6),
        ],
    )
    def test_header_phrases_are_recognised(self, text: str, scale: float) -> None:
        assert detect_scale(headers=[text]).units.scale == pytest.approx(scale)

    @pytest.mark.parametrize(
        ("text", "currency"),
        [
            ("($ in millions)", "USD"),
            ("(In thousands of Canadian dollars)", "CAD"),
            ("(in thousands of U.S. dollars)", "USD"),
            ("(in millions of €)", "EUR"),
            ("RMB'Million", "CNY"),
            ("(dollars in thousands)", "USD"),
        ],
    )
    def test_currency_is_recovered_where_stated(
        self, text: str, currency: str
    ) -> None:
        assert detect_scale(headers=[text]).units.currency == currency

    def test_a_year_before_a_magnitude_is_not_read_as_a_figure(self) -> None:
        # "2019 RMB'Million" must still declare millions; the digits belong to
        # the year, not to the magnitude.
        assert detect_scale(headers=["2019 RMB'Million"]).units.scale == pytest.approx(1e6)


class TestFalsePositives:
    @pytest.mark.parametrize(
        "text",
        [
            "The Company recorded a $5 million impairment charge.",
            "Revenue increased to $5.2 million in 2019.",
            "We repurchased 1.5 million shares during the year.",
            "A charge of 3 million was recognised.",
            "Goodwill of $1,234 million was allocated.",
        ],
    )
    def test_a_figure_in_prose_is_not_a_declaration(self, text: str) -> None:
        assert detect_scale(preceding=[text]).units.scale == pytest.approx(1.0)

    def test_prose_mentioning_a_magnitude_without_declaring_units(self) -> None:
        # "in millions" here modifies households, not the table.
        assert detect_scale(
            preceding=["The service operates in millions of households worldwide."]
        ).units.scale == pytest.approx(1.0)

    def test_a_declarative_sentence_is_still_accepted(self) -> None:
        assert detect_scale(
            preceding=["All amounts presented are in millions of dollars."]
        ).units.scale == pytest.approx(1e6)

    def test_prose_is_held_to_a_stricter_rule_than_a_header(self) -> None:
        # The same words inside the table are a header label and are trusted;
        # loose in prose, they are not.
        loose = "the group reports in millions of transactions"
        assert detect_scale(preceding=[loose]).units.scale == pytest.approx(1.0)
        assert detect_scale(headers=[loose]).units.scale == pytest.approx(1e6)


class TestPrecedence:
    def test_caption_outranks_a_header(self) -> None:
        detection = detect_scale(caption="(in thousands)", headers=["(in millions)"])
        assert detection.units.scale == pytest.approx(1e3)
        assert detection.source is ScaleSource.CAPTION

    def test_header_outranks_a_row_stem(self) -> None:
        detection = detect_scale(
            headers=["(in millions)"], row_stems=["(in thousands)"]
        )
        assert detection.source is ScaleSource.HEADER

    def test_row_stem_outranks_preceding_text(self) -> None:
        detection = detect_scale(
            row_stems=["(in thousands)"], preceding=["(in millions)"]
        )
        assert detection.source is ScaleSource.ROW_STEM

    def test_the_nearest_preceding_block_wins(self) -> None:
        # preceding is ordered nearest-first.
        detection = detect_scale(preceding=["(in thousands)", "(in millions)"])
        assert detection.units.scale == pytest.approx(1e3)

    def test_provenance_is_reported(self) -> None:
        detection = detect_scale(headers=["Year Ended 2019 (In millions)"])
        assert detection.source is ScaleSource.HEADER
        # The label is the phrase as printed, not a normalised rewrite of it,
        # so a reader can find it on the page.
        assert detection.units.scale_label == "In millions"
        assert "header" in detection.describe()


class TestThroughBuildTable:
    """The path the service actually takes."""

    def test_a_header_declaration_reaches_the_table_units(self) -> None:
        table = build(
            cells=[
                RawCell(row=0, col=0, text="($ in millions)", is_header=True),
                RawCell(row=0, col=1, text="2019", is_header=True),
                RawCell(row=1, col=0, text="Revenue"),
                RawCell(row=1, col=1, text="1,234"),
            ]
        )
        assert table.units.scale == pytest.approx(1e6)
        assert table.units.currency == "USD"

    def test_a_row_stem_declaration_reaches_the_table_units(self) -> None:
        table = build(
            cells=[
                RawCell(row=0, col=0, text="", is_header=True),
                RawCell(row=0, col=1, text="2019", is_header=True),
                RawCell(row=1, col=0, text="(In thousands)"),
                RawCell(row=1, col=1, text="1,234"),
            ]
        )
        assert table.units.scale == pytest.approx(1e3)

    def test_cells_are_scaled_but_their_text_is_untouched(self) -> None:
        table = build(
            cells=[
                RawCell(row=0, col=0, text="(in millions)", is_header=True),
                RawCell(row=0, col=1, text="2019", is_header=True),
                RawCell(row=1, col=0, text="Revenue"),
                RawCell(row=1, col=1, text="(1,234)"),
            ]
        )
        cell = table.cell_at(1, 1)
        assert cell is not None
        assert cell.text == "(1,234)", "printed text is never rewritten"
        assert cell.value is not None
        assert cell.value.num == pytest.approx(-1234.0)
        assert cell.value.scaled == pytest.approx(-1_234_000_000.0)

    def test_a_percentage_is_never_scaled(self) -> None:
        table = build(
            cells=[
                RawCell(row=0, col=0, text="(in millions)", is_header=True),
                RawCell(row=0, col=1, text="Margin", is_header=True),
                RawCell(row=1, col=0, text="Gross"),
                RawCell(row=1, col=1, text="12.3%"),
            ]
        )
        cell = table.cell_at(1, 1)
        assert cell is not None and cell.value is not None
        assert cell.value.scaled == pytest.approx(12.3), "a percentage is absolute"

    def test_no_declaration_leaves_the_table_unscaled(self) -> None:
        table = build()
        assert table.units.scale == pytest.approx(1.0)
        assert table.units.scale_label is None

    def test_the_older_nearby_text_argument_still_works(self) -> None:
        raw = RawTable(
            n_rows=2,
            n_cols=2,
            cells=(
                RawCell(row=0, col=0, text="", is_header=True),
                RawCell(row=1, col=1, text="1,234"),
            ),
            header_rows=1,
        )
        table = build_table(
            raw,
            table_id="t000",
            block_id="p0001_b000",
            page_number=1,
            section_id=None,
            bbox=None,
            nearby_text="(in thousands)",
        )
        assert table.units.scale == pytest.approx(1e3)


class TestParseScaleUnchanged:
    """`parse_scale` is the single-text entry point and keeps its contract."""

    @pytest.mark.parametrize(
        ("caption", "scale"),
        [
            ("(in millions)", 1e6),
            ("(in millions, except per share data)", 1e6),
            ("In Thousands of U.S. dollars", 1e3),
            ("(in billions)", 1e9),
            ("Amounts in trillions", 1e12),
            ("Components of income tax expense", 1.0),
            (None, 1.0),
        ],
    )
    def test_behaviour_is_preserved(self, caption: str | None, scale: float) -> None:
        assert parse_scale(caption).scale == pytest.approx(scale)


class TestCandidates:
    def test_every_declaration_in_one_string_is_returned(self) -> None:
        found = find_candidates(
            "(in millions) and (in thousands)", ScaleSource.HEADER
        )
        assert [c.scale for c in found] == [1e6, 1e3]

    def test_empty_text_yields_nothing(self) -> None:
        assert find_candidates("", ScaleSource.HEADER) == []
        assert find_candidates(None, ScaleSource.HEADER) == []
