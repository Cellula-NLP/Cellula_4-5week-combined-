"""Explain code — direct LLM call, NO RAG."""
from __future__ import annotations

import logging
from typing import AsyncIterator

from app.domain.models import StreamEvent
from app.domain.ports import LLMProvider
from app.prompts.explainer import build_explainer_prompt
from app.infrastructure.memory.session_memory import SQLiteMemoryStore

logger = logging.getLogger(__name__)


class CodeExplainer:
    def __init__(self, llm: LLMProvider, memory: SQLiteMemoryStore) -> None:
        self._llm = llm
        self._memory = memory

    async def stream(
        self,
        query: str,
        code: str | None,
        session_id: str,
    ) -> AsyncIterator[StreamEvent]:
        # Fetch the last 6 messages
        history_messages = self._memory.history(session_id, limit=6)
        history_dicts = [{"role": m.role, "content": m.content} for m in history_messages]

        messages = build_explainer_prompt(
            query=query,
            code=code,
            history=history_dicts,
            language=None,
        )

        async for token in self._llm.stream(messages, temperature=0.1):
            yield StreamEvent.token(token)