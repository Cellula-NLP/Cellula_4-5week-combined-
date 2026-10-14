"""Generate code with retrieved context (RAG)."""
from __future__ import annotations

import logging
from typing import AsyncIterator

from app.domain.models import Message, RetrievedDoc, StreamEvent
from app.domain.ports import LLMProvider, MemoryStore
from app.prompts.generator import build_generator_prompt

logger = logging.getLogger(__name__)


class CodeGenerator:
    def __init__(self, llm: LLMProvider, memory: MemoryStore) -> None:
        self._llm = llm
        self._memory = memory

    async def stream(
        self,
        query: str,
        docs: list[RetrievedDoc],
        session_id: str,
        language: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        history = self._memory.history(session_id, limit=12)
        profile = self._memory.profile(session_id)

        messages = build_generator_prompt(
            query=query,
            docs=docs,
            history=history,
            language=language or profile.get("language"),
        )

        full: list[str] = []
        async for token in self._llm.stream(messages, temperature=0.2):
            full.append(token)
            yield StreamEvent.token(token)

        # Persist the assistant turn.
        self._memory.append(
            session_id,
            Message(role="assistant", content="".join(full),
                    meta={"type": "generated_code", "docs_used": [d.id for d in docs]}),
        )