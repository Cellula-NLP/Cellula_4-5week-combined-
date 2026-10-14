"""LLM provider backed by freewhirr — a free-first router for OpenRouter.

freewhirr walks an ordered list of $0 models, failing over when one
rate-limits, errors, times out, or blows the context window.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator, Sequence

from freewhirr import Freewhirr

from app.core.config import Settings
from app.core.exceptions import LLMError
from app.domain.models import Message
from app.domain.ports import LLMProvider

logger = logging.getLogger(__name__)


class FreeWhirrClient(LLMProvider):
    """Wraps freewhirr with our domain's Message model."""

    def __init__(self, settings: Settings) -> None:
        # freewhirr reads OPENROUTER_API_KEY from the environment.
        import os
        os.environ.setdefault("OPENROUTER_API_KEY", settings.llm_api_key)
        self._client = Freewhirr()

    @staticmethod
    def _to_dicts(messages: Sequence[Message]) -> list[dict]:
        return [{"role": m.role.value, "content": m.content} for m in messages]

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        # freewhirr is sync; run it in a thread to avoid blocking the loop.
        import asyncio
        def _call() -> str:
            result = self._client.complete(messages=self._to_dicts(messages))
            return result.content
        return await asyncio.to_thread(_call)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        # freewhirr currently focuses on non-streaming; simulate stream
        # by yielding the full response once so the UI still works.
        text = await self.complete(messages, temperature=temperature, max_tokens=max_tokens)
        yield text