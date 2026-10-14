"""Pipeline ABC + shared context dataclass."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

from app.domain.models import Classification, StreamEvent


@dataclass
class PipelineContext:
    """Everything a pipeline needs besides the raw query and session id."""
    code: str | None = None
    language: str | None = None
    classification: Classification | None = None
    top_k: int = 4
    extra: dict = field(default_factory=dict)


class Pipeline(ABC):
    """A pipeline consumes a query + context and emits StreamEvents."""

    @abstractmethod
    async def run(
        self,
        *,
        session_id: str,
        query: str,
        context: PipelineContext,
    ) -> AsyncIterator[StreamEvent]:
        ...