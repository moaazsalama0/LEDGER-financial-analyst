"""Typed failures and their HTTP rendering.

Callers get one predictable body shape for every failure::

    {"error_code": "...", "message": "...", "document_id": null, "detail": {}}

``error_code`` is the field to branch on. The HTTP status tells a proxy what
to do; the code tells the orchestrator what happened.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from ledger_doc_contract.v1.enums import ErrorCode

# Written as plain integers rather than framework constants: the numbers are
# fixed by RFC 9110, while the constant names have already been renamed once
# underneath us.
HTTP_400_BAD_REQUEST = 400
HTTP_413_CONTENT_TOO_LARGE = 413
HTTP_415_UNSUPPORTED_MEDIA_TYPE = 415
HTTP_422_UNPROCESSABLE_CONTENT = 422
HTTP_500_INTERNAL_SERVER_ERROR = 500
HTTP_503_SERVICE_UNAVAILABLE = 503

#: The status each failure is reported with. Kept as one table so the error
#: contract can be read, and tested, in a single place.
STATUS_BY_CODE: dict[ErrorCode, int] = {
    ErrorCode.INVALID_PDF: HTTP_400_BAD_REQUEST,
    ErrorCode.ENCRYPTED_PDF: HTTP_400_BAD_REQUEST,
    ErrorCode.INVALID_DOCUMENT_ID: HTTP_400_BAD_REQUEST,
    ErrorCode.INVALID_OPTIONS: HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.FILE_TOO_LARGE: HTTP_413_CONTENT_TOO_LARGE,
    ErrorCode.UNSUPPORTED_MEDIA_TYPE: HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    ErrorCode.TOO_MANY_PAGES: HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.BACKEND_FAILURE: HTTP_500_INTERNAL_SERVER_ERROR,
    ErrorCode.BACKEND_UNAVAILABLE: HTTP_503_SERVICE_UNAVAILABLE,
}


class DocProcessorError(Exception):
    """Base class for every failure the service reports deliberately.

    Anything not derived from this is an unexpected bug and surfaces as a
    generic ``BACKEND_FAILURE`` rather than leaking a stack trace.
    """

    code: ErrorCode = ErrorCode.BACKEND_FAILURE

    def __init__(
        self,
        message: str,
        *,
        document_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.document_id = document_id
        self.detail = detail or {}

    @property
    def status_code(self) -> int:
        return STATUS_BY_CODE.get(self.code, HTTP_500_INTERNAL_SERVER_ERROR)

    def to_body(self) -> dict[str, Any]:
        return {
            "error_code": self.code.value,
            "message": self.message,
            "document_id": self.document_id,
            "detail": self.detail,
        }


class InvalidPDFError(DocProcessorError):
    """The upload is not a readable PDF, or contains no pages."""

    code = ErrorCode.INVALID_PDF


class EncryptedPDFError(DocProcessorError):
    """The PDF is password-protected, so its content cannot be read."""

    code = ErrorCode.ENCRYPTED_PDF


class InvalidDocumentIdError(DocProcessorError):
    """``document_id`` does not match the contract's allowed pattern."""

    code = ErrorCode.INVALID_DOCUMENT_ID


class InvalidOptionsError(DocProcessorError):
    """The ``options`` field is not valid JSON, or names something unknown."""

    code = ErrorCode.INVALID_OPTIONS


class FileTooLargeError(DocProcessorError):
    """The upload exceeds the configured size limit."""

    code = ErrorCode.FILE_TOO_LARGE


class UnsupportedMediaTypeError(DocProcessorError):
    """The upload is not declared as a PDF."""

    code = ErrorCode.UNSUPPORTED_MEDIA_TYPE


class TooManyPagesError(DocProcessorError):
    """The PDF has more pages than the service will process in one request."""

    code = ErrorCode.TOO_MANY_PAGES


class BackendFailureError(DocProcessorError):
    """The layout backend raised while parsing the document.

    Per-page failures never reach here: a page that fails is emitted with
    ``extraction_source='failed'`` and a warning, so one bad page cannot cost
    the caller the whole document. This is for a failure of the run itself.
    """

    code = ErrorCode.BACKEND_FAILURE


class BackendUnavailableError(DocProcessorError):
    """The backend exists but its models are not loaded yet."""

    code = ErrorCode.BACKEND_UNAVAILABLE


async def doc_processor_error_handler(
    _request: Request, exc: DocProcessorError
) -> JSONResponse:
    """Render a deliberate failure using the documented body shape."""
    return JSONResponse(status_code=exc.status_code, content=exc.to_body())


async def unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so an unexpected bug still returns the contract's error shape.

    The exception type is included as detail for debugging, but the message
    stays generic — a stack trace is not part of the API.
    """
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": ErrorCode.BACKEND_FAILURE.value,
            "message": "Document processing failed unexpectedly.",
            "document_id": None,
            "detail": {"exception": type(exc).__name__},
        },
    )


__all__ = [
    "STATUS_BY_CODE",
    "BackendFailureError",
    "BackendUnavailableError",
    "DocProcessorError",
    "EncryptedPDFError",
    "FileTooLargeError",
    "InvalidDocumentIdError",
    "InvalidOptionsError",
    "InvalidPDFError",
    "TooManyPagesError",
    "UnsupportedMediaTypeError",
    "doc_processor_error_handler",
    "unhandled_error_handler",
]
