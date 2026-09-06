from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings for agent-service, enforcing cloud-hosted LLM execution."""

    # LLM Engine Provider: "gemini" (default), "groq", or "mock"
    AGENT_LLM_PROVIDER: str = "gemini"

    # Cloud API Keys
    GEMINI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    # Cloud Model Selections
    GEMINI_MODEL: str = "gemini-3.6-flash"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Agent Control Parameters
    AGENT_MAX_RETRIES: int = 2
    RETRIEVAL_MOCK: bool = False
    TOP_K: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
@lru_cache
def get_settings() -> Settings:
    return Settings()