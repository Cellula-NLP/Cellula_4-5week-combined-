"""Abstract ports (interfaces) — dependency inversion boundaries."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Sequence

from app.domain.models import (
    ExecutionResult,
    Message,
    RetrievedDoc,
)


class LLMProvider(ABC):
    """Port for any chat-completion LLM."""

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return a complete (non-streamed) response."""

    @abstractmethod
    def stream(
            self,
            messages: Sequence[Message],
            *,
            temperature: float | None = None,
            max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield response tokens as they arrive."""


class EmbeddingProvider(ABC):
    """Port for text embedding."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""


class VectorStore(ABC):
    """Port for a vector database."""

    @abstractmethod
    def upsert(self, docs: Sequence[RetrievedDoc]) -> list[str]:
        """Insert or update documents. Returns their IDs."""

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        top_k: int = 4,
        filters: dict | None = None,
    ) -> list[RetrievedDoc]:
        """Return the top-k most similar documents."""

    @abstractmethod
    def count(self) -> int:
        """Total number of documents in the store."""

    @abstractmethod
    def query_by_hash(self, content_hash: str) -> list[RetrievedDoc]:
        """Find documents by content hash (deduplication)."""


class CodeRunner(ABC):
    """Port for sandboxed code execution."""

    @abstractmethod
    def run(
        self,
        code: str,
        language: str,
        *,
        timeout_s: float | None = None,
        stdin: str = "",
    ) -> ExecutionResult:
        """Execute code and return the result."""


class MemoryStore(ABC):
    """Port for conversation + profile persistence."""

    @abstractmethod
    def append(self, session_id: str, msg: Message) -> None:
        """Append a message to the session history."""

    @abstractmethod
    def history(self, session_id: str, limit: int | None = None) -> list[Message]:
        """Return the most recent messages."""

    @abstractmethod
    def profile(self, session_id: str) -> dict:
        """Return the user profile for this session."""

    @abstractmethod
    def update_profile(self, session_id: str, patch: dict) -> None:
        """Merge fields into the user profile."""

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Delete all history and profile for a session."""