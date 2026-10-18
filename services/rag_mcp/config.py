"""RAG MCP-local retrieval settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, FiniteFloat, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_RAG_INDEX_DIR = Path("services/rag_mcp/knowledge/index/uat_company_policy")
DEFAULT_QUERY_EMBEDDING_TIMEOUT = 10.0
MAX_QUERY_EMBEDDING_TIMEOUT = 60.0
DEFAULT_QUERY_EMBEDDING_MAX_RETRIES = 0
MAX_QUERY_EMBEDDING_RETRIES = 3


class RAGSettings(BaseSettings):
    """Hybrid retrieval settings owned by the RAG MCP service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    rag_index_dir: Path = DEFAULT_RAG_INDEX_DIR
    rag_candidate_multiplier: int = Field(default=3, gt=0)
    rag_max_candidates: int = Field(default=50, gt=0)
    rag_rrf_k: int = Field(default=60, gt=0)
    rag_vector_min_score: FiniteFloat = 0.0
    rag_query_embedding_timeout: FiniteFloat = Field(
        default=DEFAULT_QUERY_EMBEDDING_TIMEOUT,
        gt=0.0,
        le=MAX_QUERY_EMBEDDING_TIMEOUT,
    )
    rag_query_embedding_max_retries: int = Field(
        default=DEFAULT_QUERY_EMBEDDING_MAX_RETRIES,
        ge=0,
        le=MAX_QUERY_EMBEDDING_RETRIES,
    )

    @field_validator("rag_query_embedding_timeout", mode="before")
    @classmethod
    def _reject_boolean_timeout(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("RAG_QUERY_EMBEDDING_TIMEOUT must be a finite positive number")
        return value

    @field_validator("rag_query_embedding_max_retries", mode="before")
    @classmethod
    def _validate_embedding_max_retries(cls, value: object) -> int:
        if isinstance(value, bool):
            raise ValueError("RAG_QUERY_EMBEDDING_MAX_RETRIES must be a non-negative integer")
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized.isascii() or not normalized.isdecimal():
                raise ValueError(
                    "RAG_QUERY_EMBEDDING_MAX_RETRIES must be a non-negative integer"
                )
            return int(normalized)
        if not isinstance(value, int):
            raise ValueError("RAG_QUERY_EMBEDDING_MAX_RETRIES must be a non-negative integer")
        return value


@lru_cache(maxsize=1)
def get_rag_settings() -> RAGSettings:
    """Return the process-wide RAG configuration singleton."""
    return RAGSettings()


__all__ = [
    "DEFAULT_QUERY_EMBEDDING_MAX_RETRIES",
    "DEFAULT_QUERY_EMBEDDING_TIMEOUT",
    "DEFAULT_RAG_INDEX_DIR",
    "MAX_QUERY_EMBEDDING_RETRIES",
    "MAX_QUERY_EMBEDDING_TIMEOUT",
    "RAGSettings",
    "get_rag_settings",
]
