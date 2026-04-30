from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "EduAI API"
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    MAX_UPLOAD_SIZE_MB: int = 10

    # P3 ek ayarlar — RAG + LangGraph entegrasyonu (Task 5)
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "eduai_documents"
    # ANTHROPIC_API_KEY ENV'den otomatik (LiteLLM/AsyncAnthropic SDK çeker);
    # burada explicit field tutmuyoruz çünkü Settings'te kalsa bile asıl
    # tüketici lib'ler sys env'den okur. Sadece var olduğunu kontrol etmek
    # main.py lifespan'de yapılır.
    LLM_BACKEND: str = "anthropic"  # anthropic | qwen3-local | vllm

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
