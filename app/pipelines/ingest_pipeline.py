"""Ingest user-provided solutions (invoked from the orchestrator's FSM)."""
from __future__ import annotations

import logging
from typing import AsyncIterator

from app.domain.models import StreamEvent
from app.pipelines.base import Pipeline, PipelineContext
from app.services.ingestion import IngestionService

logger = logging.getLogger(__name__)


class IngestPipeline(Pipeline):
    def __init__(self, ingestion: IngestionService) -> None:
        self._ingestion = ingestion

    async def run(
        self,
        *,
        session_id: str,
        query: str,
        context: PipelineContext,
    ) -> AsyncIterator[StreamEvent]:
        original_query = context.extra.get("original_query", "")
        try:
            result = self._ingestion.ingest_solution(
                session_id=session_id,
                solution_text=query,
                original_query=original_query,
                language=context.language,
            )
            yield StreamEvent.message(result.message)
        except Exception as exc:
            logger.exception("Ingest pipeline failed")
            yield StreamEvent.error(f"Failed to learn from your solution: {exc}")
        yield StreamEvent.done()