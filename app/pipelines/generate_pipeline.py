"""Route 2: Generate code using RAG + relevance check.

Flow:
  query → retrieve from Chroma → relevance check
    ├─ RELEVANT     → generate code with retrieved context
    └─ NOT_RELEVANT → ask user for the correct solution
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

from app.domain.enums import ConvState, Relevance
from app.domain.models import StreamEvent, Verdict
from app.pipelines.base import Pipeline, PipelineContext
from app.services.generator import CodeGenerator
from app.services.relevance_evaluator import RelevanceEvaluator
from app.services.retriever import Retriever

logger = logging.getLogger(__name__)


class GeneratePipeline(Pipeline):
    def __init__(
        self,
        retriever: Retriever,
        evaluator: RelevanceEvaluator,
        generator: CodeGenerator,
    ) -> None:
        self._retriever = retriever
        self._evaluator = evaluator
        self._generator = generator

    async def run(
        self,
        *,
        session_id: str,
        query: str,
        context: PipelineContext,
    ) -> AsyncIterator[StreamEvent]:
        retrieval_failed = False
        try:
            docs = self._retriever.retrieve(query, top_k=context.top_k)
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            retrieval_failed = True
            docs = []

        # 2) Relevance check — only if retrieval actually worked.
        if retrieval_failed or not docs:
            # If retrieval failed (infra), let the LLM generate from training.
            # If retrieval returned nothing, we still need the relevance gate.
            if retrieval_failed:
                verdict = Verdict(Relevance.RELEVANT, 0.5, "retrieval unavailable — generating from model knowledge")
            else:
                verdict = Verdict(Relevance.NOT_RELEVANT, 0.0, "no docs retrieved")
        else:
            verdict = await self._evaluator.evaluate(query, docs)

        # 3) Branch on relevance.
        if verdict.relevance == Relevance.RELEVANT:
            async for event in self._generator.stream(
                query=query,
                docs=docs,
                session_id=session_id,
                language=context.language,
            ):
                yield event
        else:
            # Do NOT hallucinate. Ask the user for the correct solution.
            yield StreamEvent.state_change(ConvState.AWAITING_SOLUTION.value)
            msg = (
                "I couldn't find relevant knowledge for your request in my "
                "knowledge base. Could you provide the correct solution or a "
                "reference so I can learn it for future interactions?"
            )
            yield StreamEvent.message(msg)

        yield StreamEvent.done()