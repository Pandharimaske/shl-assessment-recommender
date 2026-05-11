"""
App config — loaded from environment variables via pydantic-settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    groq_small_model: str = "llama-3.1-8b-instant"
    openrouter_api_key: str | None = None
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"
    google_api_key: str | None = None
    mistral_api_key: str | None = None


    pinecone_api_key: str
    pinecone_index: str = "shl-catalog"

    hf_token: str | None = None
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"


    max_turns: int = 8
    min_context_for_recommendation: int = 2


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
