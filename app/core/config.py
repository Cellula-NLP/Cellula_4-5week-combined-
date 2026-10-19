"""Centralised, type-safe application configuration via Pydantic Settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Values come from .env or environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    tavily_api_key: str = ""

    # ── LLM ──────────────────────────────────────
    llm_provider: str = "groq"
    llm_model: str = "qwen/qwen3.8-27b"
    llm_fallback_model: str = "openai/gpt-oss-20b"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3
    llm_temperature: float = 0.2

    # ── Embeddings (code-specialized) ────────────
    embedder_model: str = "jinaai/jina-embeddings-v2-base-code"
    embedder_fallback_model: str = "BAAI/bge-small-en-v1.5"
    embedder_dim: int = 768
    embedder_query_prefix: str = ""  # Jina doesn't use a query prefix

    # ── Reranker ─────────────────────────────────
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    rerank_candidates: int = 20

    # ── Chroma ───────────────────────────────────
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "coding_knowledge"
    retrieval_top_k: int = 4
    retrieval_min_score: float = 0.25

    # ── Relevance ────────────────────────────────
    relevance_threshold: float = 0.35

    # ── Memory ───────────────────────────────────
    memory_window: int = 12
    sessions_db: str = "./data/sessions.db"

    # ── Execution ────────────────────────────────
    exec_timeout_s: float = 5.0
    exec_mem_mb: int = 256

    # ── Guardrails ───────────────────────────────
    max_input_chars: int = 20_000
    max_file_mb: int = 10
    classifier_min_confidence: float = 0.55

    @field_validator("llm_api_key")
    @classmethod
    def _warn_if_empty(cls, v: str) -> str:
        if not v or _looks_like_placeholder(v):
            import warnings
            warnings.warn(
                "LLM_API_KEY is empty or placeholder. "
                "Set a real key in .env — e.g. a free Groq key from "
                "https://console.groq.com/keys.",
                stacklevel=2,
            )
        return v

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
        Path(self.sessions_db).parent.mkdir(parents=True, exist_ok=True)


def _looks_like_placeholder(value: str) -> bool:
    upper = value.strip().upper()
    return any(marker in upper for marker in ("YOUR", "PLACEHOLDER", "XXX", "CHANGEME"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s