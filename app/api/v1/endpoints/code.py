from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
import traceback
import logging
import json

from app.api.dependencies import (
    get_code_generator, get_code_explainer, get_retriever,
    get_evaluator, get_web_search_service, get_memory_store
)
from app.domain.models import (
    CodeGenerateRequest, CodeExplainRequest,
    RetrievedDoc, Message
)
from app.domain.enums import Relevance
from app.services.generator import CodeGenerator
from app.services.explainer import CodeExplainer
from app.services.retriever import Retriever
from app.services.relevance_evaluator import RelevanceEvaluator
from app.services.web_search_service import WebSearchService
from app.infrastructure.memory.session_memory import SQLiteMemoryStore

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/explain")
async def explain_code(
        request: CodeExplainRequest,
        explainer: CodeExplainer = Depends(get_code_explainer),
        memory: SQLiteMemoryStore = Depends(get_memory_store)
):
    async def event_generator():
        try:
            # 1. Save user message
            memory.append(request.session_id, Message(role="user", content=request.query))

            # 2. Stream tokens
            full_response = []
            async for event in explainer.stream(request.query, request.code, request.session_id):
                if event.kind == "token":
                    full_response.append(event.payload)
                    # Yield JSON formatted for SSE
                    yield f"data: {json.dumps({'type': 'token', 'content': event.payload})}\n\n"

            # 3. Save AI response
            memory.append(request.session_id, Message(role="assistant", content="".join(full_response)))

            # 4. Signal completion
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as exc:
            logger.error(f"Explain stream error: {exc}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/generate")
async def generate_code(
        request: CodeGenerateRequest,
        generator: CodeGenerator = Depends(get_code_generator),
        retriever: Retriever = Depends(get_retriever),
        evaluator: RelevanceEvaluator = Depends(get_evaluator),
        web_search: WebSearchService = Depends(get_web_search_service),
        memory: SQLiteMemoryStore = Depends(get_memory_store)
):
    async def event_generator():
        try:
            # 1. Save user message
            memory.append(request.session_id, Message(role="user", content=request.query))

            # 2. Retrieve documents
            logger.info("Retrieving documents for: %s", request.query)
            docs = await run_in_threadpool(retriever.retrieve, request.query)

            # 3. Evaluate Relevance
            verdict = await evaluator.evaluate(request.query, docs)

            # Send metadata to UI (Optional but good for frontends)
            yield f"data: {json.dumps({'type': 'meta', 'verdict': verdict.relevance.value})}\n\n"

            # 4. Fallback to Web Search if not relevant
            if verdict.relevance == Relevance.NOT_RELEVANT:
                logger.info("NOT_RELEVANT. Triggering Tavily web search...")
                yield f"data: {json.dumps({'type': 'message', 'content': 'Searching Stack Overflow...'})}\n\n"

                extracted_code, detected_lang = await web_search.search_and_learn(
                    query=request.query, session_id=request.session_id
                )
                if extracted_code:
                    docs = [RetrievedDoc(
                        id="tavily_result",
                        content=extracted_code,
                        metadata={"source": "tavily_so", "language": detected_lang}
                    )]
                else:
                    docs = []

                    # 5. Stream Generated Code
            full_response = []
            async for event in generator.stream(request.query, docs, request.session_id):
                if event.kind == "token":
                    full_response.append(event.payload)
                    yield f"data: {json.dumps({'type': 'token', 'content': event.payload})}\n\n"

            # 6. Save AI response to memory
            memory.append(request.session_id, Message(role="assistant", content="".join(full_response)))

            # 7. Signal completion
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as exc:
            logger.error(f"Generate stream error: {exc}")
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")