"""Upload validation, performed before any model is asked to do work.

Everything here is cheap and refuses early. The ordering matters: size and
magic bytes are checked before the PDF is opened, so a hostile or corrupt
upload never reaches the parser.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from app.api.errors import (
    EncryptedPDFError,
    FileTooLargeError,
    InvalidDocumentIdError,
    InvalidOptionsError,
    InvalidPDFError,
    TooManyPagesError,
    UnsupportedMediaTypeError,
)
from app.core.settings import Settings
from ledger_doc_contract.v1.models import DOCUMENT_ID_PATTERN

_PDF_MAGIC = b"%PDF-"
_DOCUMENT_ID_RE = re.compile(DOCUMENT_ID_PATTERN)

#: Options the service accepts. Kept deliberately small — each one has to earn
#: its place, because every option is a variant the eval harness must cover.
_ALLOWED_OPTIONS = frozenset({"backend", "force_ocr"})

_ACCEPTED_CONTENT_TYPES = frozenset(
    {"application/pdf", "application/x-pdf", "application/octet-stream", ""}
)


def validate_content_type(content_type: str | None) -> None:
    """Reject uploads that do not claim to be a PDF.

    ``application/octet-stream`` is tolerated because many HTTP clients send
    it for any file; the magic-byte check below is what actually decides.
    """
    declared = (content_type or "").split(";")[0].strip().lower()
    if declared not in _ACCEPTED_CONTENT_TYPES:
        raise UnsupportedMediaTypeError(
            f"Expected a PDF upload, got content type {declared!r}.",
            detail={"content_type": declared, "expected": "application/pdf"},
        )


def validate_size(data: bytes, settings: Settings) -> None:
    """Reject uploads above the configured size limit."""
    if len(data) > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"File is {len(data) / 1024 / 1024:.1f} MB; the limit is "
            f"{settings.max_upload_mb} MB.",
            detail={"size_bytes": len(data), "limit_bytes": settings.max_upload_bytes},
        )


def validate_magic_bytes(data: bytes) -> None:
    """Confirm the payload really is a PDF, whatever the client declared.

    The magic string is allowed a small offset because some producers emit a
    byte-order mark or stray whitespace ahead of the header.
    """
    if not data:
        raise InvalidPDFError("The uploaded file is empty.")
    if _PDF_MAGIC not in data[:1024]:
        raise InvalidPDFError(
            "The uploaded file is not a PDF: the %PDF- header is missing.",
            detail={"first_bytes": data[:8].hex()},
        )


def resolve_document_id(document_id: str | None, data: bytes) -> str:
    """Validate a supplied id, or derive a deterministic one from the content.

    Deriving from the SHA-256 means re-ingesting the same PDF yields the same
    id, so a repeated upload updates the corpus entry instead of duplicating
    it.
    """
    if document_id is None or document_id == "":
        return f"doc_{hashlib.sha256(data).hexdigest()[:16]}"

    if not _DOCUMENT_ID_RE.match(document_id):
        raise InvalidDocumentIdError(
            "document_id may contain only letters, digits, underscore, dot, and "
            "hyphen, and must be 1-128 characters.",
            detail={"document_id": document_id, "pattern": DOCUMENT_ID_PATTERN},
        )
    return document_id


def parse_options(raw: str | None) -> dict[str, Any]:
    """Parse the optional ``options`` JSON blob.

    Unknown keys are rejected rather than ignored: a caller who misspells
    ``force_ocr`` should find out immediately, not discover weeks later that
    the flag never took effect.
    """
    if raw is None or raw.strip() == "":
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidOptionsError(
            "options must be a JSON object.", detail={"json_error": str(exc)}
        ) from exc

    if not isinstance(parsed, dict):
        raise InvalidOptionsError(
            f"options must be a JSON object, got {type(parsed).__name__}."
        )

    unknown = set(parsed) - _ALLOWED_OPTIONS
    if unknown:
        raise InvalidOptionsError(
            f"Unknown option(s): {', '.join(sorted(unknown))}.",
            detail={"unknown": sorted(unknown), "allowed": sorted(_ALLOWED_OPTIONS)},
        )

    if "force_ocr" in parsed and not isinstance(parsed["force_ocr"], bool):
        raise InvalidOptionsError("options.force_ocr must be true or false.")
    if "backend" in parsed and not isinstance(parsed["backend"], str):
        raise InvalidOptionsError("options.backend must be a string.")

    return parsed


def validate_page_count(page_count: int, settings: Settings) -> None:
    """Reject documents larger than the service will handle in one request."""
    if page_count <= 0:
        raise InvalidPDFError("The PDF contains no pages.")
    if page_count > settings.max_pages:
        raise TooManyPagesError(
            f"The PDF has {page_count} pages; the limit is {settings.max_pages}.",
            detail={"page_count": page_count, "limit": settings.max_pages},
        )


def content_sha256(data: bytes) -> str:
    """Hex SHA-256 of the source bytes, used as the cache key and provenance."""
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "EncryptedPDFError",
    "content_sha256",
    "parse_options",
    "resolve_document_id",
    "validate_content_type",
    "validate_magic_bytes",
    "validate_page_count",
    "validate_size",
]
