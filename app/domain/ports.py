"""Abstract ports (interfaces) — dependency inversion boundaries."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Sequence

from app.domain.models import (
    ExecutionResult,
    Message,
    RetrievedDoc,
    WebSearchResult,
)


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...

    @abstractmethod
    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]: ...


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, docs: Sequence[RetrievedDoc]) -> list[str]: ...

    @abstractmethod
    def query(
        self, embedding: list[float], top_k: int = 4, filters: dict | None = None
    ) -> list[RetrievedDoc]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def query_by_hash(self, content_hash: str) -> list[RetrievedDoc]: ...


class CodeRunner(ABC):
    @abstractmethod
    def run(
        self,
        code: str,
        language: str,
        *,
        timeout_s: float | None = None,
        stdin: str = "",
    ) -> ExecutionResult: ...


class MemoryStore(ABC):
    @abstractmethod
    def append(self, session_id: str, msg: Message) -> None: ...

    @abstractmethod
    def history(self, session_id: str, limit: int | None = None) -> list[Message]: ...

    @abstractmethod
    def profile(self, session_id: str) -> dict: ...

    @abstractmethod
    def update_profile(self, session_id: str, patch: dict) -> None: ...

    @abstractmethod
    def clear(self, session_id: str) -> None: ...


class WebSearchProvider(ABC):
    """Port for searching the internet."""

    @abstractmethod
    async def search(self, query: str, max_results: int = 3) -> list[WebSearchResult]: ...