"""Content-addressed cache of processed documents.

The idea is taken from thorn-nlp's precomputed-OCR sidecar, which existed
because re-running an expensive parse over the same corpus is waste. Here it
matters more: TAT-DQA is 2,758 documents, and the demo machine's GPU is shared
with the agent's language model. Parsing the corpus once offline and serving
from this cache keeps ingestion instant and the GPU free.

The key is the SHA-256 of the PDF bytes plus a *variant* — the backend's name
and the models it is configured with. The models matter as much as the name:
TableFormer's accurate and fast modes are different systems that recover
different grids, and serving one document's parse for a request that asked for
the other is precisely the unreproducible result the rest of this service goes
out of its way to prevent.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from pathlib import Path

from ledger_doc_contract.v1.models import ProcessedDocument

logger = logging.getLogger(__name__)

PROCESSING_VERSION = "2"
"""Bumped whenever the processing pipeline would assemble a different document
from the same backend parse.

History:
  1 — Phase 2 baseline.
  2 — Phase 2.1: magnitude read from headers, row stems, and a wider window of
      preceding blocks rather than the caption alone.

This is not the contract version and not the service version. It exists so an
entry written by an older build is never served to a newer one, which is what
keeps a before/after measurement honest.
"""


def cache_variant(backend_name: str, model_versions: dict[str, str]) -> str:
    """The cache namespace for one backend configuration and processing build.

    Built from the backend's own provenance rather than from a list of
    settings the cache would have to be taught about. Any backend that changes
    which models it loads reports it in ``model_versions`` already — because
    that is what a consumer needs to reproduce a result — so keying on the
    same value means a reconfiguration invalidates the cache automatically,
    for every backend, without this module knowing what a backend is.

    ``PROCESSING_VERSION`` closes the other half. What is stored here is the
    *assembled* document, not the backend's raw parse, so a change to reading
    order, heading levels, numeric parsing, or scale detection produces a
    different document from the identical bytes and the identical models.
    Without it in the key, improving the processing pipeline and then
    measuring the result silently scores the old build — the one mistake that
    would invalidate every number this service reports about itself.
    """
    canonical = json.dumps(model_versions, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:8]
    stem = backend_name if not model_versions else f"{backend_name}-{digest}"
    return f"{stem}-p{PROCESSING_VERSION}"


class DocumentCache:
    """Stores processed documents as gzipped JSON on disk."""

    def __init__(self, directory: Path, *, enabled: bool = True) -> None:
        self._directory = Path(directory)
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _path(self, content_sha256: str, variant: str) -> Path:
        # Shard by the first two hex characters so a corpus-sized run does not
        # put thousands of files in one directory.
        return self._directory / variant / content_sha256[:2] / f"{content_sha256}.json.gz"

    def get(self, content_sha256: str, variant: str) -> ProcessedDocument | None:
        """Return a cached document, or ``None`` on any miss.

        A corrupt or unreadable entry is treated as a miss and logged, never
        raised: a damaged cache should cost time, not availability.
        """
        if not self._enabled:
            return None

        path = self._path(content_sha256, variant)
        if not path.exists():
            return None

        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return ProcessedDocument.model_validate(json.load(handle))
        except Exception as exc:
            logger.warning("Discarding unreadable cache entry %s: %s", path, exc)
            return None

    def put(self, document: ProcessedDocument, variant: str) -> None:
        """Write a document to the cache, ignoring write failures.

        A cache that cannot be written is a performance problem, not a
        correctness one, so it must never fail the request that produced a
        perfectly good document.
        """
        if not self._enabled:
            return

        path = self._path(document.content_sha256, variant)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temporary file and move it into place, so a crash
            # mid-write cannot leave a half-written entry that later reads as
            # a corrupt hit.
            temporary = path.with_suffix(".tmp")
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                json.dump(document.model_dump(mode="json"), handle, sort_keys=True)
            temporary.replace(path)
        except Exception as exc:
            logger.warning("Could not write cache entry %s: %s", path, exc)


__all__ = ["DocumentCache", "cache_variant"]
