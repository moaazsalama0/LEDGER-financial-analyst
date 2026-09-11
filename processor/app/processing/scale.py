"""Finding the magnitude a financial table prints its figures in.

A filing states magnitude once and then prints bare numbers. Miss it and
``278,057`` is read as dollars when it meant 278,057,000 — a thousandfold
error carrying a correct-looking citation, which is the worst failure this
service can produce. Phase 2 measured the caption-only detector at 37.2% on
the questions whose gold declares a magnitude.

The Phase 2 corpus says where the evidence actually sits. Of the declared-
magnitude questions the caption-only detector got wrong, the phrase was
available in a **header** cell in roughly 150 cases and in a **row-label
stem** in roughly 108, against a caption that Docling leaves empty on nearly
every table. Reading only captions was looking in the one place the words
were not.

So detection reads four places, in this order of precedence:

1. ``caption`` — the table's own label. Most local and least ambiguous when
   it exists at all.
2. ``header`` — header cells and the flattened column headers. Inside the
   table, and where the convention actually puts it: ``(in millions)``,
   ``($ in millions)``, ``Year Ended December 31, 2019 (In millions)``.
3. ``row_stem`` — column-zero labels. Also inside the table, but column zero
   holds row names too, so it is a noisier place to read from than a header.
4. ``preceding_text`` — blocks just above the table, nearest first. Outside
   the table and the only source that can pick up unrelated prose, so it is
   ranked last and held to a stricter rule (see ``strict`` below).

What counts as a declaration, and what does not
-----------------------------------------------

A bare mention of "million" is not a table scale. ``$5 million`` in a
sentence is a figure. Two patterns, both anchored, are recognised:

* ``in <magnitude>`` — covers ``in millions``, ``$ in millions``,
  ``dollars in thousands``, ``amounts in thousands``, ``USD in millions``,
  ``expressed in millions``, ``in thousands of U.S. dollars``.
* ``<currency> <magnitude>`` — covers ``$ million``, ``$ MILLION``,
  ``RMB'Million``, ``EUR million``.

Both reject a magnitude word immediately preceded by a number, which is what
separates the declaration ``(in millions)`` from the figure ``$5 million``.

For ``preceding_text`` only, a match must additionally be parenthesised or
carry a declarative lead-in (``amounts``, ``dollars``, ``expressed``, a
currency symbol, and so on) within a short distance. That admits "The
following amounts are in thousands." while rejecting "we operate in millions
of households".

Provenance is returned rather than stored on the contract. Every input this
reads is already a field of ``ProcessedDocument``, so a consumer that wants
to know where a scale came from can re-derive it exactly, and v1.0 does not
have to grow a field to carry an answer it already implies.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from ledger_doc_contract.v1.models import TableUnits


class ScaleSource(str, Enum):
    """Where a magnitude declaration was read from, most reliable first."""

    CAPTION = "caption"
    HEADER = "header"
    ROW_STEM = "row_stem"
    PRECEDING_TEXT = "preceding_text"


#: Precedence order. Earlier sources win outright over later ones.
PRECEDENCE: tuple[ScaleSource, ...] = (
    ScaleSource.CAPTION,
    ScaleSource.HEADER,
    ScaleSource.ROW_STEM,
    ScaleSource.PRECEDING_TEXT,
)

MAGNITUDES: dict[str, float] = {
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
}

_MAG = r"(?P<mag>thousand|million|billion|trillion)s?"

# "in <magnitude>". The lead-in ($ / dollars / amounts / expressed) does not
# need to be in the pattern for detection; it only matters for currency and
# for the strict rule, so keeping it out avoids an unbounded alternation.
_IN_MAGNITUDE = re.compile(r"\bin\s+" + _MAG + r"\b", re.IGNORECASE)

# "<currency> <magnitude>", as in "$ million" or "RMB'Million". No digits may
# sit between the two, which is what stops "$5 million" matching here.
_CURRENCY_MAGNITUDE = re.compile(
    r"(?P<cur>[$€£¥₩₹]|\bUS\$|\b(?:USD|EUR|GBP|JPY|CNY|RMB|CAD|AUD"
    r"|CHF|HKD|SGD|INR|KRW)\b)"
    r"\s*['’]?\s*" + _MAG + r"\b",
    re.IGNORECASE,
)

# A number immediately before the magnitude word makes it a figure, not a
# declaration: "$5 million", "1.5 billion", "increased 3 million".
_TRAILING_NUMBER = re.compile(r"\d[\d,.\s]*$")

# For preceding prose only: something that marks the sentence as declaring
# units rather than merely mentioning a magnitude.
_DECLARATIVE_LEADIN = re.compile(
    r"(?:amounts?|figures?|values?|numbers?|balances?|totals?|sums?|data"
    r"|dollars?|euros?|pounds?|yen|renminbi|rupees?|won|francs?"
    r"|USD|EUR|GBP|JPY|CNY|RMB|CAD|AUD|CHF|HKD|SGD|INR|KRW"
    r"|[$€£¥₩₹]"
    r"|expressed|stated|presented|reported|shown|denominated|measured"
    r"|rounded|following|below|above)"
    r"[^.;]{0,40}$",
    re.IGNORECASE,
)

_CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₩": "KRW",
    "₹": "INR",
}

_CURRENCY_CODES = frozenset(
    {"USD", "EUR", "GBP", "JPY", "CNY", "CAD", "AUD", "CHF", "HKD", "SGD", "INR", "KRW"}
)

# Spelled-out currencies, most specific first: "Canadian dollars" must not be
# read as USD by the bare "dollars" entry below it.
_CURRENCY_WORDS: tuple[tuple[str, str], ...] = (
    ("canadian dollar", "CAD"),
    ("australian dollar", "AUD"),
    ("singapore dollar", "SGD"),
    ("hong kong dollar", "HKD"),
    ("u.s. dollar", "USD"),
    ("us dollar", "USD"),
    ("united states dollar", "USD"),
    ("pound sterling", "GBP"),
    ("dollar", "USD"),
    ("euro", "EUR"),
    ("sterling", "GBP"),
    ("renminbi", "CNY"),
    ("yuan", "CNY"),
    ("yen", "JPY"),
    ("rupee", "INR"),
    ("franc", "CHF"),
)

_CODE_RE = re.compile(r"\b([A-Z]{3})\b")


@dataclass(frozen=True)
class ScaleCandidate:
    """One magnitude declaration found in one piece of text."""

    scale: float
    label: str
    """The matched phrase, e.g. 'in millions' or "RMB'Million"."""
    currency: str | None
    source: ScaleSource
    text: str
    """The full text the phrase was found in, for diagnosis."""


@dataclass(frozen=True)
class ScaleDetection:
    """The resolved magnitude for a table, with how it was decided."""

    units: TableUnits
    source: ScaleSource | None
    candidates: tuple[ScaleCandidate, ...] = ()
    ambiguous: bool = False
    """Two candidates in the WINNING source disagreed about the magnitude.

    Conflicts between different sources are not ambiguity — that is what the
    precedence order exists to settle, deterministically. A disagreement
    inside one source is the case no rule can resolve, so it is flagged and
    the full candidate list is kept for a human to look at.
    """

    @property
    def scale(self) -> float:
        return self.units.scale

    def describe(self) -> str:
        """One line naming the decision and its evidence."""
        if self.source is None:
            return "no magnitude declared"
        flag = " AMBIGUOUS" if self.ambiguous else ""
        return (
            f"{self.units.scale:g} from {self.source.value}"
            f" ({self.units.scale_label!r}){flag}"
        )


def _normalize(text: str) -> str:
    """Fold presentation forms and collapse whitespace, without lowercasing.

    Case is preserved because currency codes are recognised by their capitals.
    """
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _inside_parentheses(text: str, position: int) -> bool:
    """True when ``position`` sits inside an unclosed '(' earlier in the text."""
    opened = text.rfind("(", 0, position)
    if opened == -1:
        return False
    return text.rfind(")", opened, position) == -1


def _currency_in(text: str) -> str | None:
    """The currency this text names, or None.

    Symbols first, then spelled-out names, then bare three-letter codes. The
    code check runs last because it is the most likely to fire on something
    that is not a currency at all.
    """
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code
    lowered = text.lower()
    for word, code in _CURRENCY_WORDS:
        if word in lowered:
            return code
    if "rmb" in lowered:
        return "CNY"
    match = _CODE_RE.search(text.upper())
    if match and match.group(1) in _CURRENCY_CODES:
        return match.group(1)
    return None


def _accepts(text: str, match: re.Match[str], *, strict: bool) -> bool:
    """Whether this magnitude match is a unit declaration rather than a figure."""
    start = match.start("mag")
    before = text[:start].rstrip()
    if _TRAILING_NUMBER.search(before):
        return False
    if not strict:
        return True
    # Prose may only declare units when it says so: parenthesised, or with a
    # lead-in word close enough to be part of the same clause.
    return _inside_parentheses(text, start) or bool(_DECLARATIVE_LEADIN.search(before))


def find_candidates(
    text: str | None, source: ScaleSource, *, strict: bool = False
) -> list[ScaleCandidate]:
    """Every magnitude declaration in one piece of text, in order."""
    if not text:
        return []
    cleaned = _normalize(text)
    if not cleaned:
        return []

    found: list[ScaleCandidate] = []
    seen: set[tuple[int, float]] = set()
    for pattern in (_IN_MAGNITUDE, _CURRENCY_MAGNITUDE):
        for match in pattern.finditer(cleaned):
            if not _accepts(cleaned, match, strict=strict):
                continue
            scale = MAGNITUDES[match.group("mag").lower()]
            key = (match.start("mag"), scale)
            if key in seen:
                continue
            seen.add(key)
            found.append(
                ScaleCandidate(
                    scale=scale,
                    label=match.group(0).strip(),
                    currency=_currency_in(cleaned),
                    source=source,
                    text=cleaned,
                )
            )
    found.sort(key=lambda candidate: cleaned.find(candidate.label))
    return found


def _collect(
    texts: Iterable[str | None], source: ScaleSource, *, strict: bool = False
) -> list[ScaleCandidate]:
    out: list[ScaleCandidate] = []
    for text in texts:
        out.extend(find_candidates(text, source, strict=strict))
    return out


def _resolve(tier: Sequence[ScaleCandidate]) -> ScaleCandidate:
    """Pick one candidate from a single source, deterministically.

    The most frequently declared magnitude wins, because a header row that
    says "(in millions)" over three column groups is three votes for the same
    thing. A tie falls back to the first in document order, which is stable
    across runs because cells arrive in row/column order.
    """
    votes = Counter(candidate.scale for candidate in tier)
    best = max(votes.values())
    winning = {scale for scale, count in votes.items() if count == best}
    for candidate in tier:
        if candidate.scale in winning:
            return candidate
    return tier[0]  # pragma: no cover - tier is never empty here


def detect_scale(
    *,
    caption: str | None = None,
    headers: Sequence[str] = (),
    row_stems: Sequence[str] = (),
    preceding: Sequence[str] = (),
) -> ScaleDetection:
    """Resolve a table's magnitude and currency from all four sources.

    ``preceding`` is ordered nearest-first: the block immediately above the
    table comes before the one above that, so the closest declaration wins.
    """
    candidates: list[ScaleCandidate] = []
    candidates += _collect([caption], ScaleSource.CAPTION)
    candidates += _collect(headers, ScaleSource.HEADER)
    candidates += _collect(row_stems, ScaleSource.ROW_STEM)
    candidates += _collect(preceding, ScaleSource.PRECEDING_TEXT, strict=True)

    for source in PRECEDENCE:
        tier = [candidate for candidate in candidates if candidate.source is source]
        if not tier:
            continue
        winner = _resolve(tier)
        currency = winner.currency or next(
            (c.currency for c in candidates if c.currency), None
        )
        return ScaleDetection(
            units=TableUnits(
                scale=winner.scale, scale_label=winner.label, currency=currency
            ),
            source=source,
            candidates=tuple(candidates),
            ambiguous=len({candidate.scale for candidate in tier}) > 1,
        )

    # No magnitude anywhere. Currency may still be declared, and reporting it
    # is what the caption-only parser already did, so that behaviour stands.
    for text in [caption, *headers, *row_stems]:
        if not text:
            continue
        currency = _currency_in(_normalize(text))
        if currency:
            return ScaleDetection(units=TableUnits(currency=currency), source=None)
    return ScaleDetection(units=TableUnits(), source=None)


__all__ = [
    "MAGNITUDES",
    "PRECEDENCE",
    "ScaleCandidate",
    "ScaleDetection",
    "ScaleSource",
    "detect_scale",
    "find_candidates",
]
