from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool
import traceback
import logging

from app.api.dependencies import (
    get_code_generator, get_code_explainer, get_retriever,
    get_evaluator, get_web_search_service, get_memory_store
)
from app.domain.models import (
    CodeGenerateRequest, CodeGenerateResponse,
    CodeExplainRequest, CodeExplainResponse,
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


@router.post("/explain", response_model=CodeExplainResponse)
async def explain_code(
        request: CodeExplainRequest,
        explainer: CodeExplainer = Depends(get_code_explainer),
        memory: SQLiteMemoryStore = Depends(get_memory_store)
):
    try:
        # 1. Save the user's prompt to memory
        memory.append(request.session_id, Message(role="user", content=request.query))

        # 2. Generate the explanation
        tokens = []
        async for event in explainer.stream(request.query, request.code, request.session_id):
            if event.kind == "token":
                tokens.append(event.payload)

        full_response = "".join(tokens)

        # 3. Save the AI's response to memory
        memory.append(request.session_id, Message(role="assistant", content=full_response))

        return CodeExplainResponse(explanation=full_response)
    except Exception as exc:
        logger.error(f"Explain error: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/generate", response_model=CodeGenerateResponse)
async def generate_code(
        request: CodeGenerateRequest,
        generator: CodeGenerator = Depends(get_code_generator),
        retriever: Retriever = Depends(get_retriever),
        evaluator: RelevanceEvaluator = Depends(get_evaluator),
        web_search: WebSearchService = Depends(get_web_search_service),
        memory: SQLiteMemoryStore = Depends(get_memory_store)
):
    try:
        # 1. Save user message
        memory.append(request.session_id, Message(role="user", content=request.query))

        # 2. Retrieve documents
        logger.info("Retrieving documents for: %s", request.query)
        docs = await run_in_threadpool(retriever.retrieve, request.query)

        # 3. Evaluate Relevance
        verdict = await evaluator.evaluate(request.query, docs)

        # 4. Fallback to Web Search if not relevant
        if verdict.relevance == Relevance.NOT_RELEVANT:
            logger.info("NOT_RELEVANT. Triggering Tavily web search...")
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

                # 5. Generate Code
        tokens = []
        async for event in generator.stream(request.query, docs, request.session_id):
            if event.kind == "token":
                tokens.append(event.payload)

        full_response = "".join(tokens)

        # 6. Save AI response to memory
        memory.append(request.session_id, Message(role="assistant", content=full_response))

        return CodeGenerateResponse(generated_text=full_response)

    except Exception as exc:
        logger.error(f"Generate error: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))