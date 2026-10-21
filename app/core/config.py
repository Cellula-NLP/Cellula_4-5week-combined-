from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── App ──
    project_name: str = "AI Backend"
    api_v1_str: str = "/api/v1"
    data_dir: str = "./data"

    # ── LLM (Shared) ──
    llm_model: str = "qwen/qwen3.8-27b"
    llm_fallback_model: str = "openai/gpt-oss-20b"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_timeout_s: float = 60.0
    llm_temperature: float = 0.2

    # ── RAG (Old) ──
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "coding_knowledge"
    embedder_model: str = "jinaai/jina-embeddings-v2-base-code"
    tavily_api_key: str = ""
    retrieval_top_k: int = 4
    retrieval_min_score: float = 0.25

    # ── Memory (NEW) ──
    sessions_db: str = "./data/sessions.db"

    # ── Voice/SQL (New) ──
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # ── Execution ──
    exec_timeout_s: float = 5.0
    exec_mem_mb: int = 256

    @field_validator("llm_api_key")
    @classmethod
    def _warn_if_empty(cls, v: str) -> str:
        if not v:
            import warnings
            warnings.warn("LLM_API_KEY is empty.", stacklevel=2)
        return v

    def ensure_dirs(self) -> None:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
        Path(self.sessions_db).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s