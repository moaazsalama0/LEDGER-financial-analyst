"""
Central configuration for orchestrator-api.

All downstream service locations, timeouts and retry behaviour are
controlled from here (and overridable via environment variables / .env)
so the service can be pointed at different hosts/ports without code changes.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Orchestrator's own bind address -------------------------------
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- Downstream service locations (per Project LEDGER contract) ----
    AGENT_SERVICE_URL: str = "http://localhost:8004"
    AGENT_QUERY_ENDPOINT: str = "/query"

    VALIDATOR_SERVICE_URL: str = "http://localhost:8006"
    VALIDATOR_ENDPOINT: str = "/validate_answer"

    PROCESSOR_SERVICE_URL: str = "http://localhost:8008"
    PROCESSOR_PROCESS_ENDPOINT: str = "/process"

    RETRIEVAL_SERVICE_URL: str = "http://localhost:8001"
    RETRIEVAL_INGEST_ENDPOINT: str = "/ingest"
    # ASSUMPTION — not in the official Contracts PDF. Confirm the real path
    # (and whether it exists at all yet) with the retrieval-api owner.
    RETRIEVAL_DOCUMENTS_ENDPOINT: str = "/documents"
    EVAL_SERVICE_URL: str = "http://localhost:8005"

    # ui-service calls the Orchestrator (not the other way around), so this
    # is registered for observability (e.g. /health) and CORS purposes only
    # — the Orchestrator makes no outbound calls to it.
    UI_SERVICE_URL: str = "http://localhost:8007"

    # --- Behaviour -------------------------------------------------------
    REQUEST_TIMEOUT_SECONDS: float = 30.0
    PROCESSOR_TIMEOUT_SECONDS: float = 900.0
    MAX_RETRIES: int = 2
    RETRY_BACKOFF_SECONDS: float = 0.6
    TOP_K: int = 5

    # --- Logging ---------------------------------------------------------
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
