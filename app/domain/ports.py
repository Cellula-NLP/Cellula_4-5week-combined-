from __future__ import annotations
from abc import ABC, abstractmethod
from typing import AsyncIterator, Sequence, Any
from app.domain.models import ExecutionResult, RetrievedDoc, WebSearchResult, ColumnSchema, Message


class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, messages: Sequence[dict], *, json_mode: bool = False,
                       temperature: float | None = None) -> str: ...

    @abstractmethod
    def stream(self, messages: Sequence[dict], *, temperature: float | None = None) -> AsyncIterator[str]: ...


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, docs: Sequence[RetrievedDoc]) -> list[str]: ...

    @abstractmethod
    def query(self, embedding: list[float], top_k: int = 4, filters: dict | None = None) -> list[RetrievedDoc]: ...

    @abstractmethod
    def query_by_hash(self, content_hash: str) -> list[RetrievedDoc]: ...


class WebSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 3) -> list[WebSearchResult]: ...


class CodeRunner(ABC):
    @abstractmethod
    def run(self, code: str, language: str, *, timeout_s: float | None = None, stdin: str = "") -> ExecutionResult: ...


class SpeechToTextProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> tuple[str, str]: ...


class DatabaseManager(ABC):
    @abstractmethod
    def create_table_from_csv(self, csv_path: str, table_name: str) -> tuple[str, int, list[ColumnSchema]]: ...

    @abstractmethod
    def get_table_schema(self, table_name: str) -> list[ColumnSchema]: ...

    @abstractmethod
    def execute_query(self, query: str) -> list[dict[str, Any]]: ...


class MemoryStore(ABC):
    """Port for conversation memory."""

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