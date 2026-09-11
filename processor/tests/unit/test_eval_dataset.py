"""Tests for loading the TAT-DQA gold files and sampling from them.

The gold record shape is fixed by a third party, so these tests double as the
written record of what this project believes that shape to be. If the mirror
ever ships a different one, this is where it surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import dataset


def write_split(
    tmp_path: Path,
    records: list[dict],
    *,
    split: str = "dev",
    pdfs: list[str] | None = None,
) -> Path:
    """Lay out a miniature dataset directory the loader can read."""
    data_dir = tmp_path / "tatdqa"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"tatdqa_dataset_{split}.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    pdf_dir = data_dir / "docs" / split
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for uid in pdfs if pdfs is not None else [r["doc"]["uid"] for r in records]:
        (pdf_dir / f"{uid}.pdf").write_bytes(b"%PDF-1.4\n")
    return data_dir


def record(uid: str, **question_overrides) -> dict:
    question = {
        "uid": f"q_{uid}",
        "order": 1,
        "question": "What was revenue?",
        "answer": ["1,234"],
        "answer_type": "span",
        "scale": "million",
        "derivation": "",
        "facts": ["1,234"],
        "req_comparison": False,
        "block_mapping": [],
    }
    question.update(question_overrides)
    return {
        "doc": {"uid": uid, "page": 1, "source": f"{uid}_2019.pdf"},
        "questions": [question],
    }


class TestLoading:
    def test_reads_documents_and_questions(self, tmp_path: Path) -> None:
        data_dir = write_split(tmp_path, [record("aaa"), record("bbb")])
        split = dataset.load_split(data_dir, "dev")

        assert split.name == "dev"
        assert len(split.documents) == 2
        assert split.question_count == 2
        assert split.documents[0].source == "aaa_2019.pdf"
        assert split.documents[0].questions[0].facts == ("1,234",)

    def test_resolves_each_document_to_its_pdf(self, tmp_path: Path) -> None:
        data_dir = write_split(tmp_path, [record("aaa")])
        document = dataset.load_split(data_dir, "dev").documents[0]
        assert document.pdf_path.name == "aaa.pdf"
        assert document.exists

    def test_a_document_without_its_pdf_is_reported_not_raised(
        self, tmp_path: Path
    ) -> None:
        # A partial download should shrink the sample, not end a run that has
        # already spent an hour parsing.
        data_dir = write_split(tmp_path, [record("aaa"), record("bbb")], pdfs=["aaa"])
        split = dataset.load_split(data_dir, "dev")
        assert [d.uid for d in split.documents] == ["aaa"]
        assert split.missing_pdfs == ("bbb",)

    def test_missing_gold_file_names_the_fix(self, tmp_path: Path) -> None:
        with pytest.raises(dataset.DatasetError, match="download_tatdqa"):
            dataset.load_split(tmp_path / "nothing", "dev")

    def test_no_pdfs_at_all_names_the_fix(self, tmp_path: Path) -> None:
        data_dir = write_split(tmp_path, [record("aaa")], pdfs=[])
        with pytest.raises(dataset.DatasetError, match="download_tatdqa"):
            dataset.load_split(data_dir, "dev")

    def test_unknown_split_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(dataset.DatasetError, match="Unknown split"):
            dataset.load_split(tmp_path, "nope")

    def test_test_split_reads_the_answered_gold_file(self) -> None:
        # The unanswered copy exists on the mirror too and would score zero
        # against everything, silently.
        assert dataset._GOLD_FILES["test"] == "tatdqa_dataset_test_gold.json"


class TestScale:
    @pytest.mark.parametrize(
        ("scale", "expected"),
        [("", 1.0), ("thousand", 1e3), ("million", 1e6), ("billion", 1e9)],
    )
    def test_scale_factor(self, scale: str, expected: float) -> None:
        question = dataset.GoldQuestion(
            uid="q", order=1, question="", answer=None, answer_type="span",
            scale=scale, derivation="", facts=(),
        )
        assert question.scale_factor == expected

    def test_percent_has_no_scale_factor(self) -> None:
        # 'percent' describes the answer's unit, not the magnitude the table
        # prints its figures in, so it must not be compared against one.
        question = dataset.GoldQuestion(
            uid="q", order=1, question="", answer=None, answer_type="arithmetic",
            scale="percent", derivation="", facts=(),
        )
        assert question.scale_factor is None


class TestSampling:
    def _documents(self, count: int) -> tuple[dataset.GoldDocument, ...]:
        return tuple(
            dataset.GoldDocument(
                uid=f"{index:03d}", source="", page=1, pdf_path=Path("x.pdf")
            )
            for index in range(count)
        )

    def test_same_seed_selects_the_same_documents(self) -> None:
        documents = self._documents(50)
        first = dataset.sample(documents, limit=10, seed=7)
        second = dataset.sample(documents, limit=10, seed=7)
        assert [d.uid for d in first] == [d.uid for d in second]

    def test_selection_does_not_depend_on_input_order(self) -> None:
        # Two backends must score the identical documents, so the sample has
        # to survive the gold file being read in a different order.
        documents = self._documents(50)
        shuffled = tuple(reversed(documents))
        assert [d.uid for d in dataset.sample(documents, limit=10, seed=1)] == [
            d.uid for d in dataset.sample(shuffled, limit=10, seed=1)
        ]

    def test_different_seeds_differ(self) -> None:
        documents = self._documents(50)
        assert [d.uid for d in dataset.sample(documents, limit=10, seed=1)] != [
            d.uid for d in dataset.sample(documents, limit=10, seed=2)
        ]

    def test_no_limit_returns_everything(self) -> None:
        documents = self._documents(5)
        assert len(dataset.sample(documents)) == 5

    def test_limit_above_the_population_returns_everything(self) -> None:
        documents = self._documents(5)
        assert len(dataset.sample(documents, limit=99)) == 5

    def test_zero_limit_returns_nothing(self) -> None:
        assert dataset.sample(self._documents(5), limit=0) == ()
