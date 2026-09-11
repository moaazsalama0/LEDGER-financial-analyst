"""Fetch the TAT-DQA corpus that the extraction harness scores against.

The brief's rule is that the original PDFs are the input to the ingestion
pipeline and the dataset's own structured content may only be used as ground
truth. This script honours that split at the filesystem level: the PDFs land
in ``docs/<split>/`` where the harness feeds them to the parser, and the gold
JSON lands beside them where only the scorer reads it.

Mirror: https://huggingface.co/datasets/next-tat/TAT-DQA (cc-by-4.0). The
project's own page links a Google Drive folder, which cannot be fetched
without a browser; the Hugging Face mirror carries the same files over plain
HTTPS, so this stays a one-command setup.

    python scripts/download_tatdqa.py --splits dev
    python scripts/download_tatdqa.py --splits dev,train,test

Downloads resume: an interrupted transfer restarts from the byte it reached
rather than from zero, which matters because the train PDFs are 1.2 GB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO = "next-tat/TAT-DQA"
BASE = "https://huggingface.co/datasets/" + REPO + "/resolve/main"

# Split -> (gold JSON on the mirror, PDF archive). The test split's questions
# ship without answers; ``_test_gold`` is the answered copy, so that is the
# one worth having.
SPLITS: dict[str, tuple[str, str]] = {
    "train": ("tatdqa_dataset_train.json", "tatdqa_docs_train.zip"),
    "dev": ("tatdqa_dataset_dev.json", "tatdqa_docs_dev.zip"),
    "test": ("tatdqa_dataset_test_gold.json", "tatdqa_docs_test.zip"),
}

_CHUNK = 1 << 20


@dataclass
class Fetched:
    """One file that is now on disk, however it got there."""

    path: Path
    size: int
    sha256: str
    skipped: bool


def _remote_size(url: str) -> int | None:
    """Bytes the server will send, or None if it will not say.

    Hugging Face serves large files from a CDN and reports the true size in
    ``x-linked-size``; ``content-length`` on the redirect describes the
    redirect body, not the file.
    """
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            headers = response.headers
    except urllib.error.URLError:
        return None
    for header in ("x-linked-size", "content-length"):
        raw = headers.get(header)
        if raw and raw.isdigit():
            return int(raw)
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(name: str, done: int, total: int | None) -> None:
    if total:
        pct = 100.0 * done / total
        line = "{:8.1f} / {:.1f} MB  {:5.1f}%".format(done / 1e6, total / 1e6, pct)
    else:
        line = "{:8.1f} MB".format(done / 1e6)
    sys.stderr.write("\r  {:<28} {}".format(name, line))
    sys.stderr.flush()


def download(url: str, target: Path, *, force: bool = False) -> Fetched:
    """Fetch ``url`` to ``target``, resuming a partial transfer if one exists."""
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = _remote_size(url)

    if target.exists() and not force:
        actual = target.stat().st_size
        if expected is None or actual == expected:
            sys.stderr.write("  {:<28} present, skipped\n".format(target.name))
            return Fetched(target, actual, _sha256(target), skipped=True)

    partial = target.with_name(target.name + ".part")
    if force:
        partial.unlink(missing_ok=True)
    start = partial.stat().st_size if partial.exists() else 0

    # A server that ignores the Range header answers 200 with the whole file,
    # so append only when it actually confirms the range with a 206.
    request = urllib.request.Request(url)
    if start:
        request.add_header("Range", "bytes={}-".format(start))

    with urllib.request.urlopen(request, timeout=120) as response:
        resuming = response.status == 206
        if not resuming:
            start = 0
        done = start
        with partial.open("ab" if resuming else "wb") as handle:
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                _progress(target.name, done, expected)
    sys.stderr.write("\n")

    if expected is not None and done != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            "{}: expected {} bytes, received {}. Re-run to retry.".format(
                target.name, expected, done
            )
        )

    partial.replace(target)
    return Fetched(target, done, _sha256(target), skipped=False)


def extract_pdfs(archive: Path, dest: Path) -> int:
    """Unpack ``archive`` into ``dest``, flattening its single top directory.

    The archives are laid out as ``<split>/<doc_uid>.pdf``. Flattening keeps
    the path the harness resolves to ``docs/<split>/<doc_uid>.pdf`` whatever
    the archive names its root.
    """
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(archive) as archive_file:
        for info in archive_file.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".pdf"):
                continue
            # Trust nothing about the member path: take the basename only, so
            # a crafted entry cannot write outside dest.
            out = dest / Path(info.filename).name
            if out.exists() and out.stat().st_size == info.file_size:
                continue
            with archive_file.open(info) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst, _CHUNK)
            written += 1
    return written


def fetch_split(split: str, data_dir: Path, *, force: bool) -> dict[str, object]:
    gold_name, docs_name = SPLITS[split]
    print("\n" + split)

    gold = download(BASE + "/" + gold_name, data_dir / gold_name, force=force)
    archive = download(
        BASE + "/" + docs_name, data_dir / "archives" / docs_name, force=force
    )

    pdf_dir = data_dir / "docs" / split
    written = extract_pdfs(archive.path, pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    print("  extracted {} new, {} PDFs total in {}".format(written, len(pdfs), pdf_dir))

    records = json.loads(gold.path.read_text(encoding="utf-8"))
    questions = sum(len(record.get("questions", ())) for record in records)
    missing = [
        record["doc"]["uid"]
        for record in records
        if not (pdf_dir / (record["doc"]["uid"] + ".pdf")).exists()
    ]
    if missing:
        print(
            "  WARNING {} documents have no PDF, e.g. {}".format(
                len(missing), missing[:3]
            )
        )
    print("  {} documents, {} questions".format(len(records), questions))

    return {
        "gold_file": gold_name,
        "gold_sha256": gold.sha256,
        "archive_sha256": archive.sha256,
        "documents": len(records),
        "questions": questions,
        "pdfs": len(pdfs),
        "missing_pdfs": len(missing),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        default="dev",
        help="Comma-separated: dev, train, test. Default dev (156 MB); train is 1.2 GB.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/tatdqa"))
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if the file is present."
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep the zips after extraction. They are 1.5 GB for all splits.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in splits if s not in SPLITS]
    if unknown:
        print(
            "unknown split(s): {}. Choose from {}".format(unknown, sorted(SPLITS)),
            file=sys.stderr,
        )
        return 2

    data_dir = args.data_dir.resolve()
    manifest_path = data_dir / "manifest.json"
    manifest: dict[str, object] = {
        "source": "https://huggingface.co/datasets/" + REPO,
        "licence": "cc-by-4.0",
        "splits": {},
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("splits", {})

    for split in splits:
        manifest["splits"][split] = fetch_split(  # type: ignore[index]
            split, data_dir, force=args.force
        )

    if not args.keep_archives:
        archives = data_dir / "archives"
        if archives.exists():
            shutil.rmtree(archives)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nmanifest {}".format(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
