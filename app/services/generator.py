"""Generate code with retrieved context (RAG)."""
from __future__ import annotations

import logging
from typing import AsyncIterator

from app.domain.enums import MessageRole
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
        # Fetch one extra turn, then drop the trailing user message if it
        # matches the current query — the orchestrator already stored it,
        # and the prompt builder re-appends `query` separately.
        raw_history = self._memory.history(session_id, limit=13)
        history = self._trim_current_turn(raw_history, query)

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
            Message(
                role=MessageRole.ASSISTANT,
                content="".join(full),
                meta={"type": "generated_code", "docs_used": [d.id for d in docs]},
            ),
        )

    # ── Internals ─────────────────────────────────

    @staticmethod
    def _trim_current_turn(
        history: list[Message],
        query: str,
    ) -> list[Message]:
        """Drop the trailing user message if it's the current query.

        The orchestrator stores the user message BEFORE dispatching to
        the pipeline, so a naive ``history`` slice would include the
        current turn. Since the prompt builder already injects `query`
        as the final user message, we must remove the duplicate.
        """
        if not history:
            return history
        last = history[-1]
        if last.role == MessageRole.USER:
            return history[:-1]
        return history