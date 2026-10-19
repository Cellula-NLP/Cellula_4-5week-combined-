"""Streamlit application entry point."""
from __future__ import annotations
import logging
import streamlit as st
from app.core.config import get_settings
from app.ui.async_bridge import iter_async
from app.ui.state import init_state
from app.ui.styles import inject_styles
from app.ui.components.chat import append_message, render_history, render_user_bubble
from app.ui.components.code_block import render_reply_controls
from app.ui.components.file_uploader import consume_uploads, render_upload_popover
from app.ui.components.memory_view import render_memory_popover

logger = logging.getLogger(__name__)

_THINKING_HTML = '<div class="thinking-box"><span class="thinking-dots"><span></span><span></span><span></span></span><span class="thinking-label">{label}…</span></div>'
_CURSOR_HTML = '<span class="streaming-cursor"></span>'

_STAGE_LABELS = {
    "rewritten_query": "Analyzing request",
    "relevance": "Checking knowledge base",
    "web_search": "Searching Stack Overflow", # <-- NEW
}

def _build_orchestrator(settings):
    from app.infrastructure.llm.openrouter_client import OpenRouterClient
    from app.infrastructure.embeddings.sentence_transformer import SentenceTransformerEmbedder
    from app.infrastructure.vectorstore.chroma_store import ChromaVectorStore
    from app.infrastructure.tools.code_runner import SubprocessSandbox
    from app.infrastructure.tools.reranker import CrossEncoderReranker
    from app.infrastructure.memory.session_memory import SQLiteMemoryStore
    from app.infrastructure.loaders.chunker import CodeAwareChunker
    from app.infrastructure.web_search.tavily_client import TavilySearchClient # <-- NEW
    from app.services.intent_classifier import IntentClassifier
    from app.services.router import Router
    from app.services.retriever import Retriever
    from app.services.relevance_evaluator import RelevanceEvaluator
    from app.services.generator import CodeGenerator
    from app.services.explainer import CodeExplainer
    from app.services.ingestion import IngestionService
    from app.services.memory_manager import MemoryManager
    from app.services.orchestrator import ConversationOrchestrator
    from app.services.query_rewriter import QueryRewriter
    from app.services.code_executor import CodeExecutor
    from app.services.web_search_service import WebSearchService # <-- NEW
    from app.pipelines.explain_pipeline import ExplainPipeline
    from app.pipelines.generate_pipeline import GeneratePipeline
    from app.domain.enums import Intent

    llm = OpenRouterClient(settings)
    embedder = SentenceTransformerEmbedder(model_name=settings.embedder_model, fallback_model=settings.embedder_fallback_model, query_prefix=settings.embedder_query_prefix)
    vdb = ChromaVectorStore(settings.chroma_path, settings.chroma_collection)
    sandbox = SubprocessSandbox(settings.exec_timeout_s, settings.exec_mem_mb)
    executor = CodeExecutor(sandbox)
    memory_store = SQLiteMemoryStore(settings.sessions_db)
    chunker = CodeAwareChunker()
    reranker = CrossEncoderReranker(model_name=settings.reranker_model, enabled=settings.reranker_enabled) if settings.reranker_enabled else None

    memory = MemoryManager(memory_store, window=settings.memory_window)
    classifier = IntentClassifier(llm, settings.classifier_min_confidence)
    retriever = Retriever(embedder, vdb, top_k=settings.retrieval_top_k, min_score=settings.retrieval_min_score, reranker=reranker, rerank_candidates=settings.rerank_candidates)
    evaluator = RelevanceEvaluator(llm, settings.relevance_threshold)
    generator = CodeGenerator(llm, memory_store)
    explainer = CodeExplainer(llm, memory_store)
    ingestion = IngestionService(embedder, vdb, chunker)
    rewriter = QueryRewriter(memory)

    # NEW: Web Search Wiring
    search_client = TavilySearchClient(settings.tavily_api_key)
    web_search_service = WebSearchService(search_client, llm, ingestion)

    router = (
        Router()
        .register(Intent.EXPLAIN_CODE, ExplainPipeline(explainer))
        .register(Intent.GENERATE_CODE, GeneratePipeline(retriever, evaluator, generator, rewriter, web_search_service))
    )

    state_store: dict = {}
    orchestrator = ConversationOrchestrator(settings=settings, classifier=classifier, router=router, memory=memory, ingestion=ingestion, state_store=state_store)
    return orchestrator, executor, memory, vdb

@st.cache_resource(show_spinner=False)
def get_runtime():
    return _build_orchestrator(get_settings())

def _render_header(memory, orchestrator) -> None:
    left, right = st.columns([10, 1], gap="small", vertical_alignment="center")
    with left:
        st.markdown('<div class="app-header-left"><div class="app-logo">🤖</div><div><div class="app-title">AI Coding Assistant</div><div class="app-subtitle">Explain · Generate · Run — powered by RAG</div></div></div>', unsafe_allow_html=True)
    with right:
        render_memory_popover(memory, orchestrator)
    st.markdown("<div class='header-rule'></div>", unsafe_allow_html=True)

def _render_composer() -> tuple[str, bool]:
    key = f"chat_input_{st.session_state.input_counter}"
    attach_col, _ = st.columns([1, 9], gap="small")
    with attach_col:
        render_upload_popover()
    user_input = st.text_area("Message", key=key, height=110, placeholder="Ask me to explain code or generate code…", label_visibility="collapsed")
    col_left, col_right = st.columns([8, 2])
    with col_right:
        send = st.button("Send  ↑", key=f"send_{st.session_state.input_counter}", type="primary", use_container_width=True)
    with col_left:
        pending = st.session_state.get("uploads") or []
        if pending:
            st.caption(f"📎 Attached: {', '.join(u['name'] for u in pending)}")
    return user_input, send

def _render_pending_turn(display_text: str, composed_text: str, executor, turn_id: str) -> None:
    render_user_bubble(display_text)
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown(_THINKING_HTML.format(label="Thinking"), unsafe_allow_html=True)
        buffer: list[str] = []
        final_meta: dict = {}
        streaming_started = False
        orchestrator, executor, _, _ = get_runtime()
        try:
            agen = orchestrator.handle(st.session_state.session_id, composed_text)
            for event in iter_async(agen):
                if event.kind in ("token", "message"):
                    streaming_started = True
                    buffer.append(event.payload)
                    placeholder.markdown("".join(buffer) + _CURSOR_HTML, unsafe_allow_html=True)
                elif event.kind == "meta":
                    final_meta.update(event.payload)
                    stage = event.payload.get("stage")
                    if not streaming_started and stage in _STAGE_LABELS:
                        placeholder.markdown(_THINKING_HTML.format(label=_STAGE_LABELS[stage]), unsafe_allow_html=True)
                elif event.kind == "state_change":
                    st.session_state.conv_state = event.payload
                elif event.kind == "error":
                    buffer.append(f"\n\n> ⚠️ {event.payload}")
                    placeholder.markdown("".join(buffer))
                elif event.kind == "done":
                    break
        except Exception as exc:
            logger.exception("UI stream failed")
            buffer.append(f"\n\n> ❌ Unexpected error: {exc}")
            placeholder.markdown("".join(buffer))
        full_reply = "".join(buffer)
        placeholder.markdown(full_reply)
    render_reply_controls(full_reply, executor, message_id=turn_id, show_run=(final_meta.get("intent") == "generate_code"))
    append_message("assistant", full_reply, meta=final_meta)

def main() -> None:
    st.set_page_config(page_title="AI Coding Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")
    inject_styles()
    init_state()
    orchestrator, executor, memory, vdb = get_runtime()
    _render_header(memory, orchestrator)
    render_history(executor)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    user_input, send_clicked = _render_composer()
    if send_clicked:
        typed = user_input.strip()
        attached_prefix = consume_uploads()
        composed = f"{attached_prefix}\n\n{typed}".strip() if attached_prefix else typed
        if not composed:
            st.warning("Please type a message before sending.")
            return
        display_text = typed or "(explain the attached file)"
        if attached_prefix:
            display_text = f"📎 *file attached*\n\n{display_text}"
        append_message("user", display_text, meta={"composed": composed})
        st.session_state.input_counter += 1
        next_assistant_index = len(st.session_state.messages)
        turn_id = f"msg{next_assistant_index}"
        _render_pending_turn(display_text, composed, executor, turn_id)
        st.rerun()

if __name__ == "__main__":
    main()