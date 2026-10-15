"""Route 2: Generate code using RAG + relevance check.

Task-compliant flow (PDF pages 7, 9, 10):
  query → retrieve from Chroma → relevance check
    ├─ RELEVANT     → generate code grounded in retrieved context
    └─ NOT_RELEVANT → do NOT hallucinate; ask the user for the correct
                      solution so it can be ingested for future use.

Follow-up handling:
  Short follow-ups like "now add error handling" are rewritten with
  context from memory before retrieval, so the vector search works.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

from app.domain.enums import ConvState, Relevance
from app.domain.models import StreamEvent, Verdict
from app.pipelines.base import Pipeline, PipelineContext
from app.services.generator import CodeGenerator
from app.services.query_rewriter import QueryRewriter
from app.services.relevance_evaluator import RelevanceEvaluator
from app.services.retriever import Retriever

logger = logging.getLogger(__name__)


class GeneratePipeline(Pipeline):
    def __init__(
        self,
        retriever: Retriever,
        evaluator: RelevanceEvaluator,
        generator: CodeGenerator,
        rewriter: QueryRewriter,
    ) -> None:
        self._retriever = retriever
        self._evaluator = evaluator
        self._generator = generator
        self._rewriter = rewriter

    async def run(
        self,
        *,
        session_id: str,
        query: str,
        context: PipelineContext,
    ) -> AsyncIterator[StreamEvent]:
        # ── 1) Rewrite follow-ups using memory ────
        retrieval_query = self._rewriter.rewrite(
            session_id, query, language=context.language,
        )
        yield StreamEvent.meta({"stage": "rewritten_query", "value": retrieval_query})

        # ── 2) Retrieve from Chroma ───────────────
        try:
            docs = self._retriever.retrieve(retrieval_query, top_k=context.top_k)
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            yield StreamEvent.message(
                "I had trouble searching my knowledge base. "
                "I'll try to answer from general knowledge."
            )
            docs = []

        # ── 3) Relevance check ────────────────────
        if not docs:
            verdict = Verdict(Relevance.NOT_RELEVANT, 0.0, "no docs retrieved")
        else:
            verdict = await self._evaluator.evaluate(retrieval_query, docs)

        yield StreamEvent.meta({
            "stage": "relevance",
            "verdict": verdict.relevance.value,
            "score": verdict.score,
            "docs": [d.id for d in docs],
        })

        # ── 4) Branch on relevance ────────────────
        if verdict.relevance == Relevance.RELEVANT:
            logger.info(
                "RELEVANT (score=%.2f) → generating with %d docs",
                verdict.score, len(docs),
            )
            async for event in self._generator.stream(
                query=query,
                docs=docs,
                session_id=session_id,
                language=context.language,
            ):
                yield event
        else:
            logger.info(
                "NOT_RELEVANT (score=%.2f) → asking user for solution",
                verdict.score,
            )
            yield StreamEvent.state_change(ConvState.AWAITING_SOLUTION.value)
            yield StreamEvent.message(
                "I couldn't find relevant knowledge for your request in my "
                "knowledge base. Could you provide the correct solution or a "
                "reference so I can learn it for future interactions?"
            )

        yield StreamEvent.done()