"""Explain code — direct LLM call, NO RAG."""
from __future__ import annotations

import logging
from typing import AsyncIterator

from app.domain.models import Message, StreamEvent
from app.domain.ports import LLMProvider, MemoryStore
from app.prompts.explainer import build_explainer_prompt

logger = logging.getLogger(__name__)


class CodeExplainer:
    """Explains code WITHOUT any retrieval. This is a hard requirement."""

    def __init__(self, llm: LLMProvider, memory: MemoryStore) -> None:
        self._llm = llm
        self._memory = memory

    async def stream(
        self,
        query: str,
        code: str | None,
        session_id: str,
    ) -> AsyncIterator[StreamEvent]:
        history = self._memory.history(session_id, limit=12)
        profile = self._memory.profile(session_id)

        messages = build_explainer_prompt(
            query=query,
            code=code,
            history=history,
            language=profile.get("language"),
        )

        # NOTE: No retriever, no embedder, no vector store touched here.
        full: list[str] = []
        async for token in self._llm.stream(messages, temperature=0.1):
            full.append(token)
            yield StreamEvent.token(token)

        self._memory.append(
            session_id,
            Message(role="assistant", content="".join(full),
                    meta={"type": "explanation"}),
        )