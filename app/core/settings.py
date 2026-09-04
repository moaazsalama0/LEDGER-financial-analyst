"""Service configuration, read once from the environment.

Every limit the API enforces is configurable here rather than hard-coded at
the call site, so the deployment can be tightened without a code change.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_VERSION = "0.1.0"


class Settings(BaseSettings):
    """Runtime configuration, overridable by ``DOC_*`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="DOC_", env_file=".env", extra="ignore")

    backend: str = Field(
        default="docling",
        description=(
            "Which layout backend to use. Docling is the only backend this "
            "service supports in production, so this is effectively fixed; it "
            "stays configurable because the name is resolved by "
            "app.backends.registry, which fails loudly on a value it does not "
            "serve — a service silently running the wrong backend is worse "
            "than one that refuses to start."
        ),
    )

    port: int = Field(
        default=8008,
        gt=0,
        le=65535,
        description=(
            "The port `python -m app.main` serves on. uvicorn's own --port "
            "still wins when the server is started through its CLI, which is "
            "what a container or a process manager usually does; this exists "
            "so the port has one written-down home rather than living only in "
            "a command line."
        ),
    )

    max_upload_mb: int = Field(default=50, gt=0)
    max_pages: int = Field(default=100, gt=0)

    # A page whose text layer yields fewer than this many characters is
    # treated as scanned and routed to OCR. Borrowed from thorn-nlp, where the
    # same threshold separated native-text PDFs from photographed ones.
    text_density_threshold: int = Field(default=30, ge=0)

    # Pages below this OCR confidence are flagged for review rather than
    # silently trusted.
    low_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    render_dpi: int = Field(default=300, gt=0, description="Rasterisation DPI for OCR paths.")

    # -- docling backend ----------------------------------------------------
    # TableFormer's two modes are different systems, not a speed dial: they
    # recover different grids and score differently. Which one ran is recorded
    # on every document so a backend comparison stays honest.
    docling_table_mode: str = Field(default="accurate", pattern="^(accurate|fast)$")

    docling_ocr: bool = Field(
        default=True,
        description=(
            "Whether the docling pipeline may OCR. Left on so scanned pages "
            "are read rather than silently returned empty; born-digital pages "
            "still take their glyphs from the text layer."
        ),
    )

    docling_artifacts_path: Path | None = Field(
        default=None,
        description=(
            "Local model directory. Set for an air-gapped deployment so a "
            "parse never depends on reaching a model host."
        ),
    )

    docling_threads: int = Field(default=4, gt=0)

    cache_enabled: bool = Field(default=True)
    cache_dir: Path = Field(default=Path(".cache/documents"))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


__all__ = ["SERVICE_VERSION", "Settings", "get_settings"]
