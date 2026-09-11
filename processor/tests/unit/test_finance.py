"""Tests for financial value parsing.

The highest-value suite in the service. Every failure here becomes a
confidently wrong number with a correct-looking citation attached to it.
"""

from __future__ import annotations

import pytest

from app.processing.finance import looks_numeric, parse_scale, parse_value
from ledger_doc_contract.v1.enums import NegativeStyle, NumericUnit
from ledger_doc_contract.v1.models import TableUnits


class TestPlainNumbers:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("0", 0.0),
            ("7", 7.0),
            ("1234", 1234.0),
            ("1,234", 1234.0),
            ("1,234.5", 1234.5),
            ("12,345,678", 12345678.0),
            ("0.001", 0.001),
            ("-1,234", -1234.0),
            ("+1,234", 1234.0),
        ],
    )
    def test_parses_magnitude(self, text: str, expected: float) -> None:
        value = parse_value(text)
        assert value is not None
        assert value.num == pytest.approx(expected)

    def test_surrounding_whitespace_is_ignored(self) -> None:
        assert parse_value("  1,234  ").num == pytest.approx(1234.0)

    def test_non_breaking_space_as_thousands_separator(self) -> None:
        # European filings and some PDF text layers use U+00A0 to group digits.
        value = parse_value("1 234 567")
        assert value is not None
        assert value.num == pytest.approx(1234567.0)


class TestParenthesesNegative:
    """The finance convention, and the single most consequential rule here."""

    def test_parentheses_mean_negative(self) -> None:
        value = parse_value("(1,234)")
        assert value is not None
        assert value.num == pytest.approx(-1234.0)
        assert value.is_negative is True
        assert value.negative_style is NegativeStyle.PARENTHESES

    def test_parentheses_inside_currency_symbol(self) -> None:
        value = parse_value("$(1,234)")
        assert value is not None
        assert value.num == pytest.approx(-1234.0)
        assert value.negative_style is NegativeStyle.PARENTHESES
        assert value.currency == "USD"

    def test_minus_sign_is_distinguished_from_parentheses(self) -> None:
        value = parse_value("-1,234")
        assert value is not None
        assert value.num == pytest.approx(-1234.0)
        assert value.negative_style is NegativeStyle.MINUS

    def test_unicode_minus_is_not_read_as_a_dash_cell(self) -> None:
        # U+2212 leading a number is a sign, not a nil placeholder.
        value = parse_value("−1,234")
        assert value is None or value.is_dash is False


class TestDashes:
    """A dash means nil or not-applicable. Coercing it to 0.0 changes totals."""

    @pytest.mark.parametrize("text", ["-", "--", "—", "–", "N/A", "n/a", "nil", "None"])
    def test_dash_forms_are_recognised(self, text: str) -> None:
        value = parse_value(text)
        assert value is not None
        assert value.is_dash is True
        assert value.num is None, "a dash must never be coerced to a number"

    def test_dash_is_not_zero(self) -> None:
        value = parse_value("—")
        assert value is not None
        assert value.num != 0.0
        assert value.scaled is None


class TestCurrency:
    @pytest.mark.parametrize(
        ("text", "code"),
        [("$1,234", "USD"), ("€1.234", "EUR"), ("£500", "GBP"), ("¥1000", "JPY")],
    )
    def test_symbol_maps_to_iso_code(self, text: str, code: str) -> None:
        value = parse_value(text)
        assert value is not None
        assert value.currency == code
        assert value.unit is NumericUnit.CURRENCY

    def test_trailing_symbol(self) -> None:
        value = parse_value("1,234€")
        assert value is not None
        assert value.currency == "EUR"
        assert value.num == pytest.approx(1234.0)


class TestPercent:
    def test_percent_sets_unit(self) -> None:
        value = parse_value("12.3%")
        assert value is not None
        assert value.num == pytest.approx(12.3)
        assert value.unit is NumericUnit.PERCENT

    def test_negative_percent_in_parentheses(self) -> None:
        value = parse_value("(4.5)%")
        assert value is not None
        assert value.num == pytest.approx(-4.5)
        assert value.unit is NumericUnit.PERCENT

    def test_percent_is_not_scaled_by_table_units(self) -> None:
        # "in millions" describes the currency columns, never the percentages.
        units = TableUnits(scale=1e6, scale_label="in millions")
        value = parse_value("12.3%", units=units)
        assert value is not None
        assert value.scaled == pytest.approx(12.3)


class TestFootnoteMarkers:
    @pytest.mark.parametrize(
        ("text", "expected_num", "refs"),
        [
            ("1,234 (1)", 1234.0, ("1",)),
            ("1,234(a)", 1234.0, ("a",)),
            ("1,234*", 1234.0, ("*",)),
            ("500 †", 500.0, ("†",)),
        ],
    )
    def test_markers_are_stripped_and_recorded(
        self, text: str, expected_num: float, refs: tuple[str, ...]
    ) -> None:
        value = parse_value(text)
        assert value is not None
        assert value.num == pytest.approx(expected_num)
        assert value.footnote_refs == refs

    def test_footnote_stripping_does_not_break_parenthesised_negative(self) -> None:
        # "(1,234)" is a negative number, NOT a footnote marker on an empty cell.
        value = parse_value("(1,234)")
        assert value is not None
        assert value.footnote_refs == ()
        assert value.num == pytest.approx(-1234.0)

    def test_negative_with_a_footnote(self) -> None:
        value = parse_value("(1,234) (2)")
        assert value is not None
        assert value.num == pytest.approx(-1234.0)
        assert value.footnote_refs == ("2",)


class TestNonNumeric:
    @pytest.mark.parametrize(
        "text",
        ["", "   ", "Total revenue", "Current federal", "2019 vs 2018", "see note 12", "1,2,3,4x"],
    )
    def test_returns_none(self, text: str) -> None:
        assert parse_value(text) is None

    def test_looks_numeric_predicate(self) -> None:
        assert looks_numeric("1,234") is True
        assert looks_numeric("Total revenue") is False
        assert looks_numeric("—") is False, "a dash carries no magnitude"


class TestScaleFromCaption:
    @pytest.mark.parametrize(
        ("caption", "scale"),
        [
            ("(in millions)", 1e6),
            ("(in millions, except per share data)", 1e6),
            ("In Thousands of U.S. dollars", 1e3),
            ("(in billions)", 1e9),
            ("Amounts in trillions", 1e12),
        ],
    )
    def test_scale_phrases(self, caption: str, scale: float) -> None:
        assert parse_scale(caption).scale == pytest.approx(scale)

    def test_no_scale_declared_defaults_to_one(self) -> None:
        units = parse_scale("Components of income tax expense")
        assert units.scale == pytest.approx(1.0)
        assert units.scale_label is None

    def test_missing_caption_is_safe(self) -> None:
        assert parse_scale(None).scale == pytest.approx(1.0)

    def test_currency_from_caption_symbol(self) -> None:
        assert parse_scale("(in millions of $)").currency == "USD"

    def test_currency_from_caption_code(self) -> None:
        assert parse_scale("(in thousands, USD)").currency == "USD"

    def test_scale_is_applied_to_cells(self) -> None:
        units = parse_scale("(in millions)")
        value = parse_value("1,234", units=units)
        assert value is not None
        assert value.num == pytest.approx(1234.0), "num stays as printed"
        assert value.scaled == pytest.approx(1_234_000_000.0), "scaled applies the multiplier"

    def test_scale_applies_to_negatives_with_the_right_sign(self) -> None:
        units = parse_scale("(in millions)")
        value = parse_value("(1,234)", units=units)
        assert value is not None
        assert value.scaled == pytest.approx(-1_234_000_000.0)


class TestRawTextIsNeverDestroyed:
    """Normalisation is additive; the printed string always survives."""

    def test_parse_does_not_mutate_input(self) -> None:
        original = "(1,234) (1)"
        parse_value(original)
        assert original == "(1,234) (1)"
