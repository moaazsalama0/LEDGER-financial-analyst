"""
Central configuration for ui-service (Project LEDGER).

The UI never talks to agent-service, retrieval-api, doc-processor-api, or
answer-validator-api directly. Per the system architecture, the UI's ONLY
downstream dependency is orchestrator-api — every question and every
document upload goes through it. Keeping this file as the single place
that knows the orchestrator's address makes it trivial to point the UI at
a different host/port (e.g. when running services on different machines)
without touching any other file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- ui-service's own bind address --------------------------------
    HOST: str = "0.0.0.0"
    PORT: int = 8007

    # --- The only downstream dependency: orchestrator-api --------------
    ORCHESTRATOR_URL: str = "http://localhost:8000"
    ORCHESTRATOR_ASK_ENDPOINT: str = "/ask"
    ORCHESTRATOR_UPLOAD_ENDPOINT: str = "/documents/upload"
    ORCHESTRATOR_HEALTH_ENDPOINT: str = "/health"

    # --- Behaviour -------------------------------------------------------
    REQUEST_TIMEOUT_SECONDS: float = 60.0
    HEALTH_TIMEOUT_SECONDS: float = 5.0

    # --- Evidence preview (bonus feature) -------------------------------
    # DPI used when rendering a cited PDF page to an image for the
    # bounding-box highlight preview. Higher = sharper but slower.
    EVIDENCE_RENDER_DPI: int = 150

    # --- Gradio ------------------------------------------------------------
    APP_TITLE: str = "LEDGER — Financial Document Intelligence Agent"
    SHARE: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
