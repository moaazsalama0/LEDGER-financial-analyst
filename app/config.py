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

    # --- Behaviour -------------------------------------------------------
    REQUEST_TIMEOUT_SECONDS: float = 30.0
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
