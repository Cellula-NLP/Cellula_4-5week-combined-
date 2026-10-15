"""Explain code — direct LLM call, NO RAG."""
from __future__ import annotations

import logging
from typing import AsyncIterator

from app.domain.enums import MessageRole
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
        # Fetch one extra turn so we can drop the current user message
        # (stored by the orchestrator) and avoid duplicating it in the
        # prompt alongside `query`.
        raw_history = self._memory.history(session_id, limit=13)
        history = self._trim_current_turn(raw_history, query)

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
            Message(
                role=MessageRole.ASSISTANT,
                content="".join(full),
                meta={"type": "explanation"},
            ),
        )

    # ── Internals ─────────────────────────────────

    @staticmethod
    def _trim_current_turn(
        history: list[Message],
        query: str,
    ) -> list[Message]:
        """Drop the trailing user message if it's the current turn."""
        if not history:
            return history
        last = history[-1]
        if last.role == MessageRole.USER:
            return history[:-1]
        return history