"""Route 1: Explain code. MUST NOT use RAG — no retriever, no embedder, no VDB."""
from __future__ import annotations

import logging
from typing import AsyncIterator

from app.domain.models import StreamEvent
from app.pipelines.base import Pipeline, PipelineContext
from app.services.explainer import CodeExplainer

logger = logging.getLogger(__name__)


class ExplainPipeline(Pipeline):
    """Direct LLM explanation. Zero retrieval components are injected."""

    def __init__(self, explainer: CodeExplainer) -> None:
        self._explainer = explainer

    async def run(
        self,
        *,
        session_id: str,
        query: str,
        context: PipelineContext,
    ) -> AsyncIterator[StreamEvent]:
        code = context.code or ""

        if not code:
            yield StreamEvent.message(
                "I don't see any code to explain. Please paste the code you'd "
                "like me to walk through, or upload a file."
            )
            yield StreamEvent.done()
            return

        async for event in self._explainer.stream(query, code, session_id):
            yield event
        yield StreamEvent.done()