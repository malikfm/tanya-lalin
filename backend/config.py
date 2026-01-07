"""Application configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    log_level: str = "INFO"
    
    # Google Gemini API
    gemini_api_key: str = ""
    
    # LLM Configuration
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "text-embedding-004"
    embedding_dim: int = 768
    
    # ChromaDB
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "legal_chunks"
    
    # RAG Configuration
    vector_search_top_k: int = 10
    keyword_search_top_k: int = 10
    final_top_k: int = 5
    min_similarity: float = 0.3
    
    # Session Configuration
    session_ttl_hours: int = 24
    max_context_messages: int = 10
    
    # CORS Configuration
    cors_origins: str = "*"  # Comma-separated list of allowed origins

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
