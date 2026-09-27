"""Settings from .env (§8.5)."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str
    telegram_webhook_secret: str
    public_url: str = ""
    llm_base_url: str = "http://localhost:8080/v1"
    llm_model: str = "dictalm3-bureaucracy"
    qdrant_path: str = "./qdrant_data"
    embed_model: str = "intfloat/multilingual-e5-large"
    min_retrieval_score: float = 0.84  # between on-topic (0.87+) and off-topic (≤0.80) scores


settings = Settings()
