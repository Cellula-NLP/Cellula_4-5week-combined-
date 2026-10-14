"""Centralised, type-safe application configuration via Pydantic Settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Values come from .env or environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── LLM ──────────────────────────────────────
    llm_provider: str = "openrouter"
    llm_model: str = "qwen/qwen3-coder:free"
    llm_fallback_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3
    llm_temperature: float = 0.2

    # ── Embeddings ───────────────────────────────
    embedder_model: str = "BAAI/bge-small-en-v1.5"
    embedder_dim: int = 384

    # ── Chroma ───────────────────────────────────
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "coding_knowledge"
    retrieval_top_k: int = 4
    relevance_threshold: float = 0.5

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
        if not v or v.startswith("sk-or-v1-YOUR"):
            import warnings
            warnings.warn(
                "LLM_API_KEY is empty or placeholder. "
                "Get a free key at https://openrouter.ai/settings/keys",
                stacklevel=2,
            )
        return v

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
        Path(self.sessions_db).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings accessor."""
    s = Settings()
    s.ensure_dirs()
    return s