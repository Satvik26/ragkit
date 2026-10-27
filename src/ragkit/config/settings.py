from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    profile: Literal["dev", "cloud"] = "dev"
    qdrant_url: str | None = None
    database_url: str | None = None
    redis_url: str | None = None
    embedding_url: str | None = None
    embedding_model: str = "BAAI/bge-m3"
    reranker_url: str | None = None
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_provider: Literal["ollama", "openai"] = "ollama"
    ollama_url: str | None = None
    ollama_model: str = "gemma4:e2b"
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1"
    openai_base_url: str = "https://api.openai.com/v1"
    langfuse_url: str | None = None
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    auth_secret: str = ""
    corpus_dir: str | None = None
    collection: str = "chunks"
    embedding_dim: int = 256
    min_evidence_score: float = 0.35
