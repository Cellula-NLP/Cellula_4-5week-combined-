"""Domain-specific exception hierarchy."""
from __future__ import annotations


class AppError(Exception):
    """Base application error."""


# ── Shared LLM Errors ──
class LLMError(AppError):
    """LLM provider returned an error."""

class RateLimitError(LLMError):
    """LLM provider rate limit hit."""

class TimeoutError(LLMError):
    """LLM request timed out."""


# ── Old RAG Errors ──
class ConfigError(AppError):
    """Invalid or missing configuration."""

class ClassificationError(AppError):
    """LLM classifier failed to produce a usable result."""

class RetrievalError(AppError):
    """Vector store retrieval failed."""

class EvaluationError(AppError):
    """Relevance evaluation failed."""

class GenerationError(AppError):
    """LLM code generation failed."""

class ExecutionError(AppError):
    """Code execution failed at the infrastructure level."""

class IngestionError(AppError):
    """Failed to ingest a document into the vector store."""

class MemoryError(AppError):
    """Session memory operation failed."""


# ── New Voice/SQL Errors ──
class STTError(AppError):
    """Speech-to-text transcription failed."""

class SQLError(AppError):
    """SQL Generation or Execution failed."""

class DatasetError(AppError):
    """Dataset processing failed."""