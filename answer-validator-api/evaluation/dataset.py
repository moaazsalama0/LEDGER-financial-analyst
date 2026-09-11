"""Reading the TAT-DQA gold files, and pairing each record with its PDF.

The dataset ships one record per document: an identifier, the report it was
cut from, and the questions asked about it. What it does not ship — and what
an extraction harness would otherwise want — is a reference parse of the
document. That absence shapes the whole harness: there is no gold cell grid to
diff against, so scoring is done against the strings the questions actually
need. See ``evaluation.matching``.

Gold record shape, verified against ``tatdqa_dataset_dev.json``::

    {"doc": {"uid": "4d41ea…", "page": 1, "source": "quicklogic_2019.pdf"},
     "questions": [
       {"uid": "9513d7…", "order": 1, "question": "…",
        "answer": ["55%", "50%"] | 2.14 | "3",
        "answer_type": "span" | "multi-span" | "arithmetic" | "count",
        "scale": "" | "thousand" | "million" | "billion" | "percent",
        "derivation": "(4,703,830-4,605,495)/4,605,495",
        "facts": ["4,605,495", "4,703,830"],
        "req_comparison": false,
        "block_mapping": [{"<block uid>": [start, end]}]}]}

``block_mapping`` points into a reference parse that is not distributed, so
its block ids cannot be resolved and it is carried but unused.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

SPLITS: tuple[str, ...] = ("train", "dev", "test")

# The gold file each split is scored from. ``test`` ships twice on the
# mirror — once with the answers stripped for leaderboard submission, once
# with them — and only the answered copy can score anything.
_GOLD_FILES: dict[str, str] = {
    "train": "tatdqa_dataset_train.json",
    "dev": "tatdqa_dataset_dev.json",
    "test": "tatdqa_dataset_test_gold.json",
}

ANSWER_TYPES: tuple[str, ...] = ("span", "multi-span", "arithmetic", "count")

# Gold ``scale`` as a multiplier. ``percent`` is deliberately absent: it
# describes the answer's unit, not the magnitude the table's figures are
# printed in, and mapping it to a number here would invite comparing it
# against ``TableUnits.scale``, which measures a different thing.
SCALE_FACTORS: dict[str, float] = {
    "": 1.0,
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
}


class DatasetError(RuntimeError):
    """The gold files or PDFs are missing, or not shaped as expected."""


@dataclass(frozen=True)
class GoldQuestion:
    """One question, with the evidence a reader needs to answer it."""

    uid: str
    order: int
    question: str
    answer: Any
    answer_type: str
    scale: str
    derivation: str
    facts: tuple[str, ...]
    req_comparison: bool = False

    @property
    def scale_factor(self) -> float | None:
        """The multiplier the printed figures carry, or None if not stated.

        None for ``percent``, which says what the answer is rather than how
        the table is scaled.
        """
        return SCALE_FACTORS.get(self.scale)

    @property
    def needs_arithmetic(self) -> bool:
        return self.answer_type == "arithmetic"


@dataclass(frozen=True)
class GoldDocument:
    """One source document and every question asked about it."""

    uid: str
    source: str
    page: int
    pdf_path: Path
    questions: tuple[GoldQuestion, ...] = ()
    split: str = ""

    @property
    def exists(self) -> bool:
        return self.pdf_path.is_file()

    @property
    def facts(self) -> tuple[str, ...]:
        """Every gold fact across all of this document's questions."""
        return tuple(fact for question in self.questions for fact in question.facts)


@dataclass(frozen=True)
class Split:
    """A loaded split, with enough provenance to reproduce a run."""

    name: str
    documents: tuple[GoldDocument, ...]
    gold_file: Path
    missing_pdfs: tuple[str, ...] = field(default=())

    @property
    def question_count(self) -> int:
        return sum(len(document.questions) for document in self.documents)


def _question(raw: dict[str, Any]) -> GoldQuestion:
    return GoldQuestion(
        uid=str(raw["uid"]),
        order=int(raw.get("order", 0)),
        question=str(raw.get("question", "")),
        answer=raw.get("answer"),
        answer_type=str(raw.get("answer_type", "")),
        scale=str(raw.get("scale", "") or ""),
        derivation=str(raw.get("derivation", "") or ""),
        # A handful of records carry a null or non-string entry; dropping the
        # entry rather than the question keeps the rest of it scorable.
        facts=tuple(str(f) for f in (raw.get("facts") or ()) if isinstance(f, (str, int, float))),
        req_comparison=bool(raw.get("req_comparison", False)),
    )


def load_split(data_dir: Path, split: str) -> Split:
    """Load one split's gold records and resolve each document to its PDF.

    Documents whose PDF is absent are dropped from ``documents`` and named in
    ``missing_pdfs``, so a partial download degrades the sample size rather
    than crashing a long run halfway through.
    """
    if split not in _GOLD_FILES:
        raise DatasetError(f"Unknown split {split!r}. Choose from {list(SPLITS)}.")

    gold_file = Path(data_dir) / _GOLD_FILES[split]
    if not gold_file.is_file():
        raise DatasetError(
            f"{gold_file} not found. Fetch it with:\n"
            f"    python scripts/download_tatdqa.py --splits {split}"
        )

    raw_records = json.loads(gold_file.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise DatasetError(f"{gold_file} should hold a list of records.")

    pdf_dir = Path(data_dir) / "docs" / split
    documents: list[GoldDocument] = []
    missing: list[str] = []

    for record in raw_records:
        doc = record.get("doc") or {}
        uid = str(doc.get("uid", ""))
        if not uid:
            continue
        pdf_path = pdf_dir / f"{uid}.pdf"
        document = GoldDocument(
            uid=uid,
            source=str(doc.get("source", "")),
            page=int(doc.get("page", 0) or 0),
            pdf_path=pdf_path,
            questions=tuple(_question(q) for q in record.get("questions", ())),
            split=split,
        )
        if document.exists:
            documents.append(document)
        else:
            missing.append(uid)

    if not documents:
        raise DatasetError(
            f"No PDFs found under {pdf_dir} for any of the {len(raw_records)} "
            f"gold records. Fetch them with:\n"
            f"    python scripts/download_tatdqa.py --splits {split}"
        )

    return Split(
        name=split,
        documents=tuple(documents),
        gold_file=gold_file,
        missing_pdfs=tuple(missing),
    )


def sample(
    documents: Sequence[GoldDocument],
    *,
    limit: int | None = None,
    seed: int = 0,
) -> tuple[GoldDocument, ...]:
    """Take a reproducible subsample, or everything when ``limit`` is None.

    Sampling is by document rather than by question because parsing is the
    expensive step and every question about a document rides along free. The
    sample is drawn from a list sorted by uid, so the same ``(limit, seed)``
    selects the same documents whatever order the gold file happened to be in.
    """
    ordered = sorted(documents, key=lambda document: document.uid)
    if limit is None or limit >= len(ordered):
        return tuple(ordered)
    if limit <= 0:
        return ()
    chosen = random.Random(seed).sample(ordered, limit)
    return tuple(sorted(chosen, key=lambda document: document.uid))


def iter_questions(documents: Iterable[GoldDocument]) -> Iterable[tuple[GoldDocument, GoldQuestion]]:
    """Yield every ``(document, question)`` pair."""
    for document in documents:
        for question in document.questions:
            yield document, question


__all__ = [
    "ANSWER_TYPES",
    "SCALE_FACTORS",
    "SPLITS",
    "DatasetError",
    "GoldDocument",
    "GoldQuestion",
    "Split",
    "iter_questions",
    "load_split",
    "sample",
]
