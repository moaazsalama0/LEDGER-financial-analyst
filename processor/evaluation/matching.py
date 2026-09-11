"""What counts as "the parser recovered this fact", stated precisely.

TAT-DQA ships no reference parse, so there is no cell grid to diff against.
What it does ship is, per question, the literal strings a reader must find in
the document to answer it. That turns out to be the more useful target
anyway: a cell recovered but never asked about proves nothing, while a fact
the parser lost is one no amount of retrieval or reasoning can recover.

A fact is resolved to exactly one outcome, in this order of preference:

``table_cell_text``
    A table cell whose text matches the fact as printed. The strongest
    result: the figure is in the grid, with row and column labels attached,
    so ``search_tables`` can find it and the calculator tool can use it.

``table_cell_value``
    A table cell whose *parsed number* equals the fact's, though the printed
    strings differ — ``(1,234)`` against ``-1,234``, or ``$1,234`` against
    ``1,234``. Also a success, and specifically a success of the finance
    parser rather than of the layout model.

``block_text``
    A prose *phrase* found in some block's text, but in no table cell. What
    ``span`` questions mostly need.

``block_text_value``
    A *figure* found in some block's prose rather than in a cell. Retrievable
    and citable, but a number landing here usually means a table was missed
    or flattened, so it is counted apart from a cell hit.

    Figures are matched against whole parsed tokens, never as substrings.
    Substring search cannot be made safe for them: a word-edge guard treats a
    comma as a boundary, so ``181`` matches inside ``1,181,204``. Token
    matching also makes the punctuation of the gold string irrelevant, which
    it should be — a text layer that emits ``$ 278,057`` with a detached
    currency symbol has recovered the figure, and calling that a miss because
    the gold reads ``$278,057`` would report a spacing artefact as a lost
    fact.

``furniture_only``
    Found only in a block classified as a running header or footer. Counted
    as a miss, and reported loudly: those blocks are excluded from
    ``reading_order``, so the fact would never reach a retrieval chunk. When
    this number is not near zero, page-furniture demotion is eating content.

``not_found``
    Nowhere in the parse.

Where this metric can flatter a parser, stated so nobody quotes it wrongly:

* It is **recall, not precision**. A bare ``0.4`` matches any cell holding
  0.4 anywhere on the page. ``unique`` on a match records whether exactly one
  cell matched, and the report carries the unique-match rate beside the
  headline so the ambiguity is visible rather than buried.
* A **short numeric fact is easier to hit by chance** than a long text one.
  Scores are therefore broken out by answer type, since ``span`` questions
  lean on prose and ``arithmetic`` on table figures.
* It says nothing about whether the figure was **associated with the right
  label** — only that it was recovered. Row/column-label correctness needs a
  reference grid, which this dataset does not provide.

None of that undermines the comparison the harness exists for: both backends
are scored by the identical rule, so the difference between them is real even
where the absolute number is generous.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from app.processing.finance import parse_value
from ledger_doc_contract.v1.models import ProcessedDocument

# Relative tolerance for calling two parsed figures the same number. Tight
# enough that 1.234 and 1.235 stay distinct, loose enough to absorb float
# repr noise from two different parse paths.
_REL_TOL = 1e-9


class Outcome(str, Enum):
    """Where a gold fact was found, best result first."""

    TABLE_CELL_TEXT = "table_cell_text"
    TABLE_CELL_VALUE = "table_cell_value"
    BLOCK_TEXT = "block_text"
    BLOCK_TEXT_VALUE = "block_text_value"
    FURNITURE_ONLY = "furniture_only"
    NOT_FOUND = "not_found"

    @property
    def found(self) -> bool:
        """True when the fact is reachable by the retrieval service.

        Furniture is excluded deliberately: those blocks never enter
        ``reading_order``, so a fact found only there is lost in practice.
        """
        return self in (
            Outcome.TABLE_CELL_TEXT,
            Outcome.TABLE_CELL_VALUE,
            Outcome.BLOCK_TEXT,
            Outcome.BLOCK_TEXT_VALUE,
        )

    @property
    def in_table(self) -> bool:
        return self in (Outcome.TABLE_CELL_TEXT, Outcome.TABLE_CELL_VALUE)


@dataclass(frozen=True)
class FactMatch:
    """The verdict on one gold fact."""

    fact: str
    outcome: Outcome
    numeric: bool
    table_id: str | None = None
    cell_id: str | None = None
    block_id: str | None = None
    page_number: int | None = None
    candidates: int = 0
    """How many places matched. 1 is an unambiguous hit; >1 is a weaker one."""

    @property
    def found(self) -> bool:
        return self.outcome.found

    @property
    def unique(self) -> bool:
        return self.found and self.candidates == 1


def normalize(text: str) -> str:
    """Fold a string to the form both sides of a text comparison use.

    NFKC first, because a PDF text layer and the gold file disagree about
    which of several Unicode spaces, dashes, and quote marks to use for the
    same glyph. Then case folding and whitespace collapse, so a match does
    not turn on a line break the extractor happened to preserve.
    """
    folded = unicodedata.normalize("NFKC", text)
    # Map every dash-like codepoint onto the ASCII hyphen. Reports mix these
    # freely and the gold file does not always copy the same one.
    folded = folded.translate(_DASH_TABLE)
    return " ".join(folded.casefold().split())


_DASH_TABLE = {ord(ch): "-" for ch in "‐‑‒–—―−"}


# A non-numeric fact worth searching for as free text. Very short strings
# ("a", "of") appear inside longer words constantly, so a substring search for
# them means nothing. Numeric facts do not need this guard: they are matched
# against whole parsed tokens, which cannot hit inside a longer number.
_MIN_TEXT_FACT = 4

_WORD_EDGE = re.compile(r"\w", re.UNICODE)

# Candidate figures inside a sentence. Deliberately generous — anything it
# picks up is handed to the finance parser, which is the authority on whether
# it is really a number, so a loose pattern costs a parse call and never a
# wrong match.
_NUMBER_TOKEN = re.compile(
    r"\(?\s?[$€£¥₩₹]?\s?[+-]?\d[\d,.  ]*\d|\(?[$€£¥₩₹]?\s?[+-]?\d"
)


@lru_cache(maxsize=4096)
def _numeric(fact: str) -> float | None:
    """The number this fact denotes, or None when it is not a bare figure.

    Uses the service's own finance parser rather than a second
    implementation. That is deliberate: the harness should score the parser
    the service actually ships, and a scorer with its own private notion of
    what ``(1,234)`` means would be measuring the difference between two
    parsers instead of the difference between two backends.
    """
    value = parse_value(fact)
    return None if value is None else value.num


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=_REL_TOL, abs_tol=0.0)


def _contains_phrase(haystack: str, needle: str) -> bool:
    """Substring search that will not match inside a longer word or number.

    Without the edge check, the fact ``181`` matches ``1,181,204`` and the
    fact ``net`` matches ``network``, both of which would silently inflate
    recall.
    """
    if not needle:
        return False
    start = haystack.find(needle)
    while start != -1:
        before = haystack[start - 1] if start else ""
        after_index = start + len(needle)
        after = haystack[after_index] if after_index < len(haystack) else ""
        starts_clean = not (before and _WORD_EDGE.match(before))
        ends_clean = not (after and _WORD_EDGE.match(after))
        if starts_clean and ends_clean:
            return True
        start = haystack.find(needle, start + 1)
    return False


@dataclass(frozen=True)
class _Blocks:
    """One collection of blocks, indexed both ways a fact can be looked up."""

    texts: tuple[tuple[str, str, int], ...] = ()
    """(normalized text, block_id, page)."""

    values: tuple[tuple[float, str, int], ...] = ()
    """(parsed num, block_id, page) for every figure written in the prose."""


@dataclass(frozen=True)
class _Index:
    """Everything in one parsed document, arranged for repeated lookup.

    Built once per document. A document's questions share it, which is what
    keeps scoring negligible next to parsing.
    """

    cell_text: dict[str, tuple[tuple[str, str, int], ...]]
    """normalized cell text -> ((table_id, cell_id, page), ...)"""

    cell_value: tuple[tuple[float, str, str, int], ...]
    """(parsed num, table_id, cell_id, page), for numeric comparison."""

    body: _Blocks
    """Blocks in reading order — everything retrieval will actually see."""

    furniture: _Blocks
    """Blocks excluded from reading order, kept to detect content lost to them."""


def _numbers_in(text: str) -> list[float]:
    """Every figure a sentence contains, as the finance parser reads them."""
    found: list[float] = []
    for match in _NUMBER_TOKEN.finditer(text):
        value = parse_value(match.group(0))
        if value is not None and value.num is not None:
            found.append(value.num)
    return found


def _search(blocks: _Blocks, needle: str, number: float | None) -> tuple[str, int, int] | None:
    """Find ``needle`` in ``blocks``; returns (block_id, page, hit count).

    A numeric fact is matched against whole parsed tokens and never as a
    substring. Substring search cannot be made safe for figures: the word-edge
    guard treats a comma as a boundary, so ``181`` matches inside
    ``1,181,204`` and recall becomes fiction. Token matching has no such hole,
    because the tokenizer takes the whole number before the parser sees it.
    """
    if number is not None:
        hits = [entry for entry in blocks.values if _same_number(entry[0], number)]
        if hits:
            return hits[0][1], hits[0][2], len(hits)
        return None

    if len(needle) < _MIN_TEXT_FACT:
        return None
    hits = [entry for entry in blocks.texts if _contains_phrase(entry[0], needle)]
    if hits:
        return hits[0][1], hits[0][2], len(hits)
    return None


def build_index(document: ProcessedDocument) -> _Index:
    """Arrange a parsed document for fact lookup."""
    cell_text: dict[str, list[tuple[str, str, int]]] = {}
    cell_value: list[tuple[float, str, str, int]] = []

    for table in document.tables:
        for cell in table.cells:
            key = normalize(cell.text)
            if key:
                cell_text.setdefault(key, []).append(
                    (table.table_id, cell.cell_id, table.page_number)
                )
            if cell.value is not None and cell.value.num is not None:
                cell_value.append(
                    (cell.value.num, table.table_id, cell.cell_id, table.page_number)
                )

    in_reading_order = {block.block_id for block in document.iter_blocks()}
    texts: dict[bool, list[tuple[str, str, int]]] = {True: [], False: []}
    values: dict[bool, list[tuple[float, str, int]]] = {True: [], False: []}

    for page in document.pages:
        for block in page.blocks:
            body = block.block_id in in_reading_order
            text = normalize(block.text)
            if not text:
                continue
            texts[body].append((text, block.block_id, block.page_number))
            for number in _numbers_in(text):
                values[body].append((number, block.block_id, block.page_number))

    return _Index(
        cell_text={key: tuple(value) for key, value in cell_text.items()},
        cell_value=tuple(cell_value),
        body=_Blocks(texts=tuple(texts[True]), values=tuple(values[True])),
        furniture=_Blocks(texts=tuple(texts[False]), values=tuple(values[False])),
    )


def match_fact(fact: str, index: _Index) -> FactMatch:
    """Resolve one gold fact against a parsed document."""
    needle = normalize(fact)
    number = _numeric(fact)
    is_numeric = number is not None

    if not needle:
        return FactMatch(fact=fact, outcome=Outcome.NOT_FOUND, numeric=is_numeric)

    # 1. A cell printed exactly as the gold says.
    hits = index.cell_text.get(needle)
    if hits:
        table_id, cell_id, page = hits[0]
        return FactMatch(
            fact=fact,
            outcome=Outcome.TABLE_CELL_TEXT,
            numeric=is_numeric,
            table_id=table_id,
            cell_id=cell_id,
            page_number=page,
            candidates=len(hits),
        )

    # 2. A cell holding the same number, written differently.
    if number is not None:
        numeric_hits = [
            entry for entry in index.cell_value if _same_number(entry[0], number)
        ]
        if numeric_hits:
            _, table_id, cell_id, page = numeric_hits[0]
            return FactMatch(
                fact=fact,
                outcome=Outcome.TABLE_CELL_VALUE,
                numeric=True,
                table_id=table_id,
                cell_id=cell_id,
                page_number=page,
                candidates=len(numeric_hits),
            )

    # 3. In the prose the retrieval service will actually see. A figure is
    #    reported as BLOCK_TEXT_VALUE and a phrase as BLOCK_TEXT, because the
    #    two are found by different means and mean different things.
    hit = _search(index.body, needle, number)
    if hit is not None:
        block_id, page, count = hit
        return FactMatch(
            fact=fact,
            outcome=Outcome.BLOCK_TEXT_VALUE if is_numeric else Outcome.BLOCK_TEXT,
            numeric=is_numeric,
            block_id=block_id,
            page_number=page,
            candidates=count,
        )

    # 4. Present, but only where the chunker will never look.
    hit = _search(index.furniture, needle, number)
    if hit is not None:
        block_id, page, count = hit
        return FactMatch(
            fact=fact,
            outcome=Outcome.FURNITURE_ONLY,
            numeric=is_numeric,
            block_id=block_id,
            page_number=page,
            candidates=count,
        )

    return FactMatch(fact=fact, outcome=Outcome.NOT_FOUND, numeric=is_numeric)


# -- scale ------------------------------------------------------------------


class ScaleVerdict(str, Enum):
    """How the table's detected magnitude compares with the gold answer's.

    Getting this wrong is a thousandfold error in the answer, with a correct
    citation attached, which is the most dangerous failure this service can
    produce. It earns its own metric rather than hiding inside fact recall.
    """

    CORRECT = "correct"
    MISSED = "missed"
    """Gold declares a scale; the parser found none. Answers come out 1000x small."""

    SPURIOUS = "spurious"
    """The parser declared a scale the gold does not. Answers come out 1000x large."""

    WRONG = "wrong"
    """Both declare one, and they differ — thousands read as millions."""

    UNSCORABLE = "unscorable"
    """No table backs this question, or the gold scale is 'percent'."""


def score_scale(
    gold_scale: str,
    expected_factor: float | None,
    table_scales: tuple[float, ...],
) -> ScaleVerdict:
    """Compare the gold magnitude against the tables a question's facts hit.

    ``table_scales`` holds the detected scale of every table a fact of this
    question matched. A question whose facts landed in no table cannot be
    scored: the failure is already counted as a fact miss, and double
    counting it here would make two independent-looking metrics move together.

    With facts spread across several tables, agreement by any one of them is
    accepted — the figures the answer needs may legitimately come from a
    table whose neighbours are scaled differently.
    """
    if expected_factor is None or gold_scale == "percent":
        return ScaleVerdict.UNSCORABLE
    if not table_scales:
        return ScaleVerdict.UNSCORABLE

    if any(_same_number(scale, expected_factor) for scale in table_scales):
        return ScaleVerdict.CORRECT

    declared = [scale for scale in table_scales if not _same_number(scale, 1.0)]
    if expected_factor == 1.0:
        return ScaleVerdict.SPURIOUS
    if not declared:
        return ScaleVerdict.MISSED
    return ScaleVerdict.WRONG


__all__ = [
    "FactMatch",
    "Outcome",
    "ScaleVerdict",
    "build_index",
    "match_fact",
    "normalize",
    "score_scale",
]
