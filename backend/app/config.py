"""Validated application configuration without import-time side effects."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool | None = None

    openai_api_key: SecretStr = SecretStr("")
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, ge=1)
    analysis_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "none"
    generation_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "low"
    ai_request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    ai_max_retries: int = Field(default=2, ge=0, le=10)

    corpus_dir: Path = REPOSITORY_ROOT / "corpus"
    static_dir: Path = REPOSITORY_ROOT / "frontend" / "dist"

    vector_search_top_k: int = Field(default=20, ge=1, le=50)
    bm25_search_top_k: int = Field(default=20, ge=1, le=50)
    fusion_top_k: int = Field(default=20, ge=1, le=50)
    final_evidence_top_k: int = Field(default=5, ge=1, le=20)
    min_vector_relevance: float = Field(default=0.3, ge=0.0, le=1.0)
    vector_rrf_weight: float = Field(default=1.0, gt=0)
    bm25_rrf_weight: float = Field(default=1.0, gt=0)
    article_reference_rrf_weight: float = Field(default=1.5, gt=0)
    retrieval_profile: Literal[
        "legacy", "balanced", "balanced_article", "balanced_article_diverse"
    ] = "balanced_article"

    reranker_enabled: bool = True
    reranker_model: str = "gpt-4o-mini"
    reranker_candidate_top_k: int = Field(default=10, ge=2, le=20)
    output_validation_enabled: bool = True
    output_validation_model: str = "gpt-4o-mini"

    redis_url: str = ""
    session_ttl_seconds: int = Field(default=86_400, ge=60)
    session_max_messages: int = Field(default=40, ge=2, le=200)
    max_context_messages: int = Field(default=10, ge=0, le=40)

    chat_rate_limit: str = "10/minute"
    history_rate_limit: str = "60/minute"
    delete_rate_limit: str = "20/minute"
    cors_origins: str = ""

    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_project: str = "tanya-lalin"
    langsmith_tracing: bool = False

    model_config = SettingsConfigDict(
        env_file=REPOSITORY_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("corpus_dir", "static_dir", mode="before")
    @classmethod
    def _expand_path(cls, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
        return path.resolve()

    @property
    def json_logs_enabled(self) -> bool:
        if self.log_json is not None:
            return self.log_json
        return self.environment == "production"

    @property
    def parsed_cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def openai_key(self) -> str:
        return self.openai_api_key.get_secret_value()

    @property
    def embedding_provider(self) -> str:
        """Return the provider identity recorded in corpus manifests."""

        return "openai"

    @property
    def langsmith_key(self) -> str:
        return self.langsmith_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process configuration."""

    return Settings()
