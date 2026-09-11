"""Backend selection by name, with lazy construction.

Two stances carried over from thorn-nlp, both earned the hard way there:

* **Construct lazily.** Importing this module must not pull a model into
  memory, so the service starts fast and the test suite imports freely.
* **Fail loud on a name it does not know.** A service silently running a
  different backend than the operator intended produces results nobody can
  reproduce, which is worse than a service that refuses to start.
"""

from __future__ import annotations

from typing import Callable

from app.api.errors import BackendUnavailableError
from app.backends.base import LayoutBackend

_FACTORIES: dict[str, Callable[[], LayoutBackend]] = {}
_INSTANCES: dict[str, LayoutBackend] = {}


def register(name: str, factory: Callable[[], LayoutBackend]) -> None:
    """Register a backend factory under a name.

    The service registers only what it supports in production. This is public
    so the evaluation harness can add the frozen Phase 2 baseline for a
    historical rerun, and so a test can inject a fake — neither of which the
    running service ever does.
    """
    _FACTORIES[name] = factory


def available() -> tuple[str, ...]:
    """Names that :func:`get_backend` will accept."""
    return tuple(sorted(_FACTORIES))


def get_backend(name: str) -> LayoutBackend:
    """Return the singleton backend registered under ``name``.

    Raises :class:`BackendUnavailableError` for an unknown name, listing what
    is available so the mistake is obvious from the response alone.
    """
    if name not in _FACTORIES:
        raise BackendUnavailableError(
            f"Unknown backend {name!r}.",
            detail={"requested": name, "available": list(available())},
        )

    if name not in _INSTANCES:
        try:
            _INSTANCES[name] = _FACTORIES[name]()
        except ImportError as exc:
            # A backend whose optional dependency is missing is a
            # configuration problem, reported as such rather than as a crash.
            raise BackendUnavailableError(
                f"Backend {name!r} is registered but its dependencies are not "
                f"installed.",
                detail={"backend": name, "import_error": str(exc)},
            ) from exc

    return _INSTANCES[name]


def reset() -> None:
    """Restore the registry to what the service ships with.

    Constructed instances are dropped *and* any extra registration is undone,
    so a backend one test or tool added cannot leak into the next and make the
    service look like it offers something it does not. For tests; never called
    in serving.
    """
    _INSTANCES.clear()
    _FACTORIES.clear()
    _register_builtins()


def _register_builtins() -> None:
    """Register the backends that ship with the service.

    Docling is the whole list. The seam stays because the API layer must not
    care which model reads a page — that is what let the Phase 2 comparison
    happen at all — but an entry here is a claim that the service will serve
    that backend in production, and only one backend does.

    The factory imports inside the function body so docling's dependencies are
    only required by a caller that actually selects it.
    """

    def _docling() -> LayoutBackend:
        from app.backends.docling_backend import DoclingBackend
        from app.core.settings import get_settings

        settings = get_settings()
        return DoclingBackend(
            table_mode=settings.docling_table_mode,
            ocr_enabled=settings.docling_ocr,
            artifacts_path=(
                str(settings.docling_artifacts_path)
                if settings.docling_artifacts_path
                else None
            ),
            num_threads=settings.docling_threads,
        )

    register("docling", _docling)


_register_builtins()

__all__ = ["available", "get_backend", "register", "reset"]
