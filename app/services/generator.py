"""Generate code with retrieved context (RAG)."""
from __future__ import annotations

import logging
from typing import AsyncIterator

from app.domain.models import RetrievedDoc, StreamEvent
from app.domain.ports import LLMProvider
from app.prompts.generator import build_generator_prompt
from app.infrastructure.memory.session_memory import SQLiteMemoryStore

logger = logging.getLogger(__name__)


class CodeGenerator:
    def __init__(self, llm: LLMProvider, memory: SQLiteMemoryStore) -> None:
        self._llm = llm
        self._memory = memory

    async def stream(
        self,
        query: str,
        docs: list[RetrievedDoc],
        session_id: str,
        language: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        # Fetch the last 6 messages from the session memory
        history_messages = self._memory.history(session_id, limit=6)

        # Convert dataclass Messages to dicts for the LLM
        history_dicts = [{"role": m.role, "content": m.content} for m in history_messages]

        messages = build_generator_prompt(
            query=query,
            docs=docs,
            history=history_dicts,
            language=language,
        )

        async for token in self._llm.stream(messages, temperature=0.2):
            yield StreamEvent.token(token)