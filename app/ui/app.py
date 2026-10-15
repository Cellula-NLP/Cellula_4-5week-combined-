"""Streamlit application entry point — Claude-like teal UI."""
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


# ── Thinking indicator + streaming cursor (inline HTML) ─────────

_THINKING_HTML = (
    '<div class="thinking-box">'
    '<span class="thinking-dots"><span></span><span></span><span></span></span>'
    '<span class="thinking-label">{label}…</span>'
    '</div>'
)

_CURSOR_HTML = '<span class="streaming-cursor"></span>'

# Friendly labels for internal pipeline stages (from StreamEvent.meta).
_STAGE_LABELS = {
    "rewritten_query": "Analyzing request",
    "relevance": "Checking knowledge base",
}


# ── Runtime wiring ──────────────────────────────────────

def _build_orchestrator(settings):
    from app.infrastructure.llm.openrouter_client import OpenRouterClient
    from app.infrastructure.embeddings.sentence_transformer import SentenceTransformerEmbedder
    from app.infrastructure.vectorstore.chroma_store import ChromaVectorStore
    from app.infrastructure.tools.code_runner import SubprocessSandbox
    from app.infrastructure.memory.session_memory import SQLiteMemoryStore
    from app.infrastructure.loaders.chunker import CodeAwareChunker
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
    from app.pipelines.explain_pipeline import ExplainPipeline
    from app.pipelines.generate_pipeline import GeneratePipeline
    from app.domain.enums import Intent

    llm = OpenRouterClient(settings)
    embedder = SentenceTransformerEmbedder(settings.embedder_model)
    vdb = ChromaVectorStore(settings.chroma_path, settings.chroma_collection)

    sandbox = SubprocessSandbox(settings.exec_timeout_s, settings.exec_mem_mb)
    executor = CodeExecutor(sandbox)

    memory_store = SQLiteMemoryStore(settings.sessions_db)
    chunker = CodeAwareChunker()

    memory = MemoryManager(memory_store, window=settings.memory_window)
    classifier = IntentClassifier(llm, settings.classifier_min_confidence)
    retriever = Retriever(embedder, vdb, top_k=settings.retrieval_top_k)
    evaluator = RelevanceEvaluator(llm, settings.relevance_threshold)
    generator = CodeGenerator(llm, memory_store)
    explainer = CodeExplainer(llm, memory_store)
    ingestion = IngestionService(embedder, vdb, chunker)
    rewriter = QueryRewriter(memory)

    router = (
        Router()
        .register(Intent.EXPLAIN_CODE, ExplainPipeline(explainer))
        .register(
            Intent.GENERATE_CODE,
            GeneratePipeline(retriever, evaluator, generator, rewriter),
        )
    )

    state_store: dict = {}
    orchestrator = ConversationOrchestrator(
        settings=settings,
        classifier=classifier,
        router=router,
        memory=memory,
        ingestion=ingestion,
        state_store=state_store,
    )
    return orchestrator, executor, memory, vdb


@st.cache_resource(show_spinner=False)
def get_runtime():
    settings = get_settings()
    return _build_orchestrator(settings)


# ── Header ──────────────────────────────────────────────

def _render_header(vdb, memory) -> None:
    """Header: logo + title on the left, memory popover on the right.

    The chunk count is intentionally NOT shown — it's internal state
    that a user doesn't need to see in a production app.
    """
    left, right = st.columns([10, 1], gap="small", vertical_alignment="center")
    with left:
        st.markdown(
            """
            <div class="app-header-left">
                <div class="app-logo">🤖</div>
                <div>
                    <div class="app-title">AI Coding Assistant</div>
                    <div class="app-subtitle">Explain · Generate · Run — powered by RAG</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        render_memory_popover(memory)
    st.markdown("<div class='header-rule'></div>", unsafe_allow_html=True)


# ── Composer ────────────────────────────────────────────

def _render_composer() -> tuple[str, bool]:
    key = f"chat_input_{st.session_state.input_counter}"
    attach_col, _ = st.columns([1, 9], gap="small")
    with attach_col:
        render_upload_popover()
    user_input = st.text_area(
        "Message",
        key=key,
        height=110,
        placeholder=(
            "Ask me to explain code or generate code…  "
            "(Enter for a new line · Ctrl/Cmd + Enter to send)"
        ),
        label_visibility="collapsed",
    )
    col_left, col_right = st.columns([8, 2])
    with col_right:
        send = st.button(
            "Send  ↑",
            key=f"send_{st.session_state.input_counter}",
            type="primary",
            use_container_width=True,
        )
    with col_left:
        pending = st.session_state.get("uploads") or []
        if pending:
            names = ", ".join(u["name"] for u in pending)
            st.caption(f"📎 Attached: {names}")
    return user_input, send


# ── Turn rendering ──────────────────────────────────────

def _render_pending_turn(
    display_text: str,
    composed_text: str,
    executor,
    turn_id: str,
) -> None:
    """Stream the assistant's reply with a live thinking indicator + cursor.

    States:
      1. Thinking — animated dots with a friendly stage label.
      2. Streaming — tokens appear with a blinking cursor at the end.
      3. Done — cursor removed, full Markdown rendered cleanly.
    """
    render_user_bubble(display_text)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()

        # State 1 — Thinking
        placeholder.markdown(
            _THINKING_HTML.format(label="Thinking"),
            unsafe_allow_html=True,
        )

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
                    # State 2 — Streaming with cursor
                    placeholder.markdown(
                        "".join(buffer) + _CURSOR_HTML,
                        unsafe_allow_html=True,
                    )
                elif event.kind == "meta":
                    final_meta.update(event.payload)
                    stage = event.payload.get("stage")
                    # Only update the thinking label if streaming hasn't started
                    if not streaming_started and stage in _STAGE_LABELS:
                        placeholder.markdown(
                            _THINKING_HTML.format(label=_STAGE_LABELS[stage]),
                            unsafe_allow_html=True,
                        )
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

        # State 3 — Done (no cursor)
        full_reply = "".join(buffer)
        placeholder.markdown(full_reply)

    # Render Run buttons for THIS turn using the deterministic turn_id.
    render_reply_controls(full_reply, executor, message_id=turn_id)

    append_message("assistant", full_reply, meta=final_meta)


# ── Main ────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="AI Coding Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    init_state()

    orchestrator, executor, memory, vdb = get_runtime()

    _render_header(vdb, memory)

    # History — pass executor so past Run buttons reappear with stable keys.
    render_history(executor)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    user_input, send_clicked = _render_composer()

    if send_clicked:
        typed = user_input.strip()
        attached_prefix = consume_uploads()
        composed = (
            f"{attached_prefix}\n\n{typed}".strip()
            if attached_prefix else typed
        )
        if not composed:
            st.warning("Please type a message before sending.")
            return

        display_text = typed or "(explain the attached file)"
        if attached_prefix:
            display_text = f"📎 *file attached*\n\n{display_text}"

        append_message("user", display_text, meta={"composed": composed})
        st.session_state.input_counter += 1

        # The turn_id must match what render_history uses on the next rerun.
        next_assistant_index = len(st.session_state.messages)
        turn_id = f"msg{next_assistant_index}"

        _render_pending_turn(display_text, composed, executor, turn_id)
        st.rerun()


if __name__ == "__main__":
    main()