"""FastAPI application for doc-processor-api.

Service 2 of the LEDGER pipeline: raw financial PDF in, structured document
contract out. It knows nothing about retrieval, reasoning, or answers.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.errors import (
    DocProcessorError,
    doc_processor_error_handler,
    unhandled_error_handler,
)
from app.api.routes import router
from app.backends import registry
from app.core.settings import SERVICE_VERSION, get_settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("doc_processor")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Resolve and warm the configured backend before serving.

    Warming at startup rather than on the first request means the readiness
    probe tells the truth, and the first caller does not pay the model load —
    which for the Docling backend is a download measured in minutes.

    Two failures are treated differently on purpose, because they are
    different problems:

    * **An unknown backend name fails here, at startup.** It is a
      misconfiguration, and a service quietly running a different backend than
      the operator intended produces results nobody can reproduce.
    * **Models that will not load do not stop the process.** That is an
      environment problem — no weights on disk, no route to the model host —
      and the service is more useful up and reporting ``/ready: false`` with
      the reason than dead in a restart loop. A request then fails with
      ``BACKEND_FAILURE`` saying so, rather than with a connection refused
      that tells the orchestrator nothing.
    """
    settings = get_settings()
    backend = registry.get_backend(settings.backend)

    try:
        backend.warm()
    except Exception as exc:
        logger.error(
            "doc-processor-api %s started but backend %r could not load its "
            "models: %s. /ready will report false until this is resolved.",
            SERVICE_VERSION,
            backend.name,
            exc,
        )
        yield
        return

    logger.info(
        "doc-processor-api %s ready (backend=%s, available=%s)",
        SERVICE_VERSION,
        backend.name,
        ", ".join(registry.available()),
    )
    yield


def create_app() -> FastAPI:
    """Build the application. A factory so tests can construct it in isolation."""
    app = FastAPI(
        title="LEDGER doc-processor-api",
        version=SERVICE_VERSION,
        summary="Converts raw financial PDFs into the LEDGER document contract.",
        lifespan=lifespan,
    )
    app.include_router(router)
    app.add_exception_handler(DocProcessorError, doc_processor_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
    return app


app = create_app()


def main() -> None:
    """Serve on the configured port.

    A second way in, for a local run or a deployment that would rather set
    ``DOC_PORT`` than assemble a command line::

        python -m app.main

    ``uvicorn app.main:app --port 8008`` still works and still wins. This path
    reads the same settings object the service itself reads, so the port is
    written down once rather than living only in a shell history.

    The application object is passed rather than the ``"app.main:app"`` import
    string: under ``python -m`` this module is already loaded as ``__main__``,
    and the string would import it a second time under its real name, building
    a second application and warming the backend twice.
    """
    import uvicorn

    settings = get_settings()
    logger.info("serving doc-processor-api on port %d", settings.port)
    uvicorn.run(app, port=settings.port)


if __name__ == "__main__":
    main()

__all__ = ["app", "create_app", "main"]
