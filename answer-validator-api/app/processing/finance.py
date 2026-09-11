"""Parsing of financial figures as they are actually printed in reports.

This is the layer the old thorn-nlp pipeline has no equivalent of, and the one
that decides whether an arithmetic question can be answered at all. A table
cell reading ``(1,234)`` is negative one thousand two hundred thirty-four; a
parser that reads it as positive produces a confidently wrong answer with a
correct-looking citation attached.

Design rules:

* **Additive only.** Nothing here mutates the cell's raw text. The parse is
  attached beside it, so a downstream disagreement is always recoverable.
* **Refuse rather than guess.** Text that is not clearly a number yields
  ``None``, not ``0.0``. A dash is recorded as a dash, never coerced to zero,
  because summing dashes as zeros silently changes a total.
* **Scale is a table property, not a cell one.** Reports declare magnitude
  once in the caption and then print bare numbers, so the multiplier is read
  from the caption by :func:`parse_scale` and applied to every cell.
"""

from __future__ import annotations

import re
import unicodedata

from ledger_doc_contract.v1.enums import NegativeStyle, NumericUnit
from ledger_doc_contract.v1.models import NumericValue, TableUnits

# Currency symbols we can map to an ISO 4217 code with confidence. A bare
# "$" is ambiguous across USD/CAD/AUD; TAT-DQA is US-filing dominated and the
# reports state their currency in the caption, so USD is the pragmatic
# default and the caption parse can override it.
_CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "USD",
    "€": "EUR",  # euro
    "£": "GBP",  # pound
    "¥": "JPY",  # yen
    "₩": "KRW",  # won
    "₹": "INR",  # rupee
    "₽": "RUB",  # ruble
    "₣": "FRF",
}

_CURRENCY_CODES = frozenset(
    {"USD", "EUR", "GBP", "JPY", "CNY", "CAD", "AUD", "CHF", "HKD", "SGD", "INR", "KRW"}
)

# Every character that means "nothing here". Includes the ASCII hyphen, the
# Unicode en/em dashes and minus sign, and the common textual placeholders.
_DASH_CHARS = frozenset("-‐‑‒–—―−")
_DASH_WORDS = frozenset({"n/a", "na", "nil", "none", "not applicable", "—", "–"})

# Scale phrases, longest first so "in thousands of millions" cannot match the
# shorter phrase by accident.
_SCALE_PHRASES: tuple[tuple[str, float], ...] = (
    ("trillions", 1e12),
    ("trillion", 1e12),
    ("billions", 1e9),
    ("billion", 1e9),
    ("millions", 1e6),
    ("million", 1e6),
    ("thousands", 1e3),
    ("thousand", 1e3),
)

# A trailing footnote marker: "1,234 (1)", "1,234 (a)", "1,234*", "1,234 †".
# Deliberately anchored to the END so it cannot eat a parenthesised negative,
# which wraps the whole number rather than trailing it.
_TRAILING_FOOTNOTE_RE = re.compile(
    r"\s*(?:\((?P<paren>[0-9a-z]{1,3})\)|(?P<sym>[*†‡§]{1,3}))\s*$",
    re.IGNORECASE,
)

# The numeric core: optional sign, digits with optional group separators,
# optional decimal part. Requires at least one digit.
_NUMBER_RE = re.compile(
    r"^[+-]?(?:\d{1,3}(?:[,  \s]\d{3})+|\d+)(?:\.\d+)?$"
)

_CODE_RE = re.compile(r"\b([A-Z]{3})\b")


def _normalize_text(text: str) -> str:
    """Fold Unicode presentation forms and collapse whitespace.

    NFKC turns full-width digits and the various non-breaking spaces used as
    thousands separators into their ASCII equivalents, which is what lets one
    regex handle text lifted from a PDF text layer and from OCR alike.
    """
    folded = unicodedata.normalize("NFKC", text)
    return " ".join(folded.split())


def _is_dash(text: str) -> bool:
    """True when the cell means 'nil or not applicable' rather than a number."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.lower().strip(" .") in _DASH_WORDS:
        return True
    # A run made up only of dash characters, e.g. "-", "--", "—".
    return all(ch in _DASH_CHARS for ch in stripped)


def parse_scale(caption: str | None) -> TableUnits:
    """Read the magnitude and currency one piece of text declares.

    Financial tables say ``(in millions, except per share data)`` once and
    then print bare numbers. Recovering that multiplier is the difference
    between answering 1,234 and 1,234,000,000.

    Returns default units (scale 1.0, no currency) when the text is absent or
    declares nothing — never guesses a scale.

    Kept as the single-text entry point; it delegates to
    :func:`app.processing.scale.detect_scale`, which is the one implementation
    of what a magnitude phrase looks like. A table with headers and
    surrounding context to read should call that directly rather than joining
    its sources into a string, because precedence between them is the part
    that matters and a joined string throws it away.
    """
    from app.processing.scale import detect_scale

    return detect_scale(caption=caption).units


def parse_value(
    text: str,
    *,
    units: TableUnits | None = None,
) -> NumericValue | None:
    """Parse one table cell into a :class:`NumericValue`, or ``None``.

    ``None`` means "this cell is not a number" — a row label, a blank, a
    sentence. That is different from a dash, which IS returned as a
    ``NumericValue`` carrying ``is_dash=True`` and ``num=None``, because the
    distinction between "no value printed here" and "this is not a numeric
    column" matters when deciding whether a total can be computed.

    Recognised forms::

        (1,234)     -> -1234.0, negative_style=parentheses
        1,234.5     ->  1234.5
        $1,234      ->  1234.0, currency=USD, unit=currency
        12.3%       ->  12.3,   unit=percent
        1,234 (1)   ->  1234.0, footnote_refs=("1",)
        -           ->  None,   is_dash=True
    """
    if text is None:
        return None

    units = units or TableUnits()
    raw = _normalize_text(text)
    if not raw:
        return None

    if _is_dash(raw):
        return NumericValue(
            is_dash=True,
            currency=units.currency,
            unit=NumericUnit.CURRENCY if units.currency else None,
        )

    working = raw
    footnotes: list[str] = []

    # Strip trailing footnote markers, possibly several ("1,234 (1)(2)").
    while True:
        match = _TRAILING_FOOTNOTE_RE.search(working)
        if not match:
            break
        marker = match.group("paren") or match.group("sym")
        # A trailing "(1)" on a bare "(1)" cell is the number, not a footnote.
        candidate = working[: match.start()].strip()
        if not candidate:
            break
        footnotes.insert(0, marker)
        working = candidate

    # Parenthesised negative, the finance convention: (1,234) == -1234.
    is_paren_negative = False
    if working.startswith("(") and working.endswith(")"):
        inner = working[1:-1].strip()
        if inner:
            is_paren_negative = True
            working = inner

    # Percent.
    unit: NumericUnit | None = None
    if working.endswith("%"):
        unit = NumericUnit.PERCENT
        working = working[:-1].strip()

    # Currency symbol, leading or trailing.
    currency: str | None = None
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if working.startswith(symbol):
            currency = code
            working = working[len(symbol) :].strip()
            break
        if working.endswith(symbol):
            currency = code
            working = working[: -len(symbol)].strip()
            break

    # A parenthesised negative can sit inside the currency symbol: $(1,234).
    if not is_paren_negative and working.startswith("(") and working.endswith(")"):
        inner = working[1:-1].strip()
        if inner:
            is_paren_negative = True
            working = inner

    if not working or not _NUMBER_RE.match(working):
        return None

    # Remove group separators, then read the magnitude.
    digits = re.sub(r"[,  \s]", "", working)
    try:
        num = float(digits)
    except ValueError:  # pragma: no cover - guarded by _NUMBER_RE
        return None

    negative_style: NegativeStyle | None = None
    if is_paren_negative:
        num = -abs(num)
        negative_style = NegativeStyle.PARENTHESES
    elif num < 0:
        negative_style = NegativeStyle.MINUS

    if unit is None:
        if currency or units.currency:
            unit = NumericUnit.CURRENCY
            currency = currency or units.currency

    # Percentages are already absolute; scaling them by "in millions" would
    # be nonsense, so the table multiplier applies to currency and counts only.
    scaled = num if unit is NumericUnit.PERCENT else num * units.scale

    return NumericValue(
        num=num,
        scaled=scaled,
        currency=currency,
        unit=unit,
        is_negative=num < 0,
        negative_style=negative_style,
        is_dash=False,
        footnote_refs=tuple(footnotes),
    )


def looks_numeric(text: str) -> bool:
    """Cheap predicate for deciding whether a column is a numeric column."""
    value = parse_value(text)
    return value is not None and value.num is not None


__all__ = ["looks_numeric", "parse_scale", "parse_value"]
