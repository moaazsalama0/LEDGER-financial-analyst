from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings for agent-service, enforcing cloud-hosted LLM execution."""

    # LLM Engine Provider: "hybrid" (Gemini then Groq), "gemini", "groq", or "mock"
    AGENT_LLM_PROVIDER: str = "hybrid"

    # Cloud API Keys
    GEMINI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    # Cloud Model Selections
    GEMINI_MODEL: str = "gemini-3.6-flash"
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    # Agent Control Parameters
    AGENT_MAX_RETRIES: int = 2
    RETRIEVAL_MOCK: bool = False
    RETRIEVAL_API_URL: str = "http://localhost:8001"
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
