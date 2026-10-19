"""Route 2: Generate code using RAG + Tavily Web Search fallback."""
from __future__ import annotations

import logging
from typing import AsyncIterator

from app.domain.enums import Relevance
from app.domain.models import StreamEvent, Verdict, RetrievedDoc
from app.pipelines.base import Pipeline, PipelineContext
from app.services.generator import CodeGenerator
from app.services.query_rewriter import QueryRewriter
from app.services.relevance_evaluator import RelevanceEvaluator
from app.services.retriever import Retriever
from app.services.web_search_service import WebSearchService

logger = logging.getLogger(__name__)


class GeneratePipeline(Pipeline):
    def __init__(
        self,
        retriever: Retriever,
        evaluator: RelevanceEvaluator,
        generator: CodeGenerator,
        rewriter: QueryRewriter,
        web_search: WebSearchService,
    ) -> None:
        self._retriever = retriever
        self._evaluator = evaluator
        self._generator = generator
        self._rewriter = rewriter
        self._web_search = web_search

    async def run(
        self,
        *,
        session_id: str,
        query: str,
        context: PipelineContext,
    ) -> AsyncIterator[StreamEvent]:
        # 1) Rewrite follow-ups
        retrieval_query = self._rewriter.rewrite(
            session_id, query, language=context.language,
        )
        yield StreamEvent.meta({"stage": "rewritten_query", "value": retrieval_query})

        # 2) Retrieve from Chroma
        retrieval_failed = False
        docs: list = []
        try:
            docs = self._retriever.retrieve(retrieval_query, top_k=context.top_k)
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            retrieval_failed = True

        # 2a) Retrieval failure → degrade gracefully
        if retrieval_failed:
            yield StreamEvent.meta({"stage": "relevance", "verdict": "unknown", "reason": "retrieval_failed"})
            yield StreamEvent.message("I couldn't reach my knowledge base just now, so I'll answer from general knowledge instead.\n\n")
            async for event in self._generator.stream(query=query, docs=[], session_id=session_id, language=context.language):
                yield event
            yield StreamEvent.done()
            return

        # 3) Relevance check
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

        # 4) Branch on relevance
        if verdict.relevance == Relevance.NOT_RELEVANT:
            logger.info("NOT_RELEVANT → Triggering Stack Overflow search.")
            yield StreamEvent.meta({"stage": "web_search", "query": query})

            # Call Tavily and Ingest to Chroma (The Learning Loop)
            extracted_code, detected_lang = await self._web_search.search_and_learn(
                query=retrieval_query, session_id=session_id
            )

            if extracted_code:
                # We found a solution! Create a doc to feed the generator.
                docs = [RetrievedDoc(
                    id="tavily_result",
                    content=extracted_code,
                    metadata={"source": "tavily_so", "language": detected_lang}
                )]
                verdict = Verdict(Relevance.RELEVANT, 1.0, "Found via Stack Overflow")
            else:
                # Web search failed entirely
                yield StreamEvent.message(
                    "I searched Stack Overflow but couldn't find a reliable code solution. "
                    "I'll generate an answer from my general knowledge.\n\n"
                )
                async for event in self._generator.stream(query=query, docs=[], session_id=session_id, language=context.language):
                    yield event
                yield StreamEvent.done()
                return

        # 5) Generate Final Code
        logger.info("Generating code with %d docs.", len(docs))
        async for event in self._generator.stream(
            query=query,
            docs=docs,
            session_id=session_id,
            language=context.language,
        ):
            yield event

        yield StreamEvent.done()