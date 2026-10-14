"""Streamlit application entry point."""
from __future__ import annotations

import logging

import streamlit as st

from app.core.config import get_settings
from app.ui.async_bridge import iter_async
from app.ui.state import init_state, reset_session
from app.ui.styles import inject_styles
from app.ui.components.chat import append_message, render_history
from app.ui.components.code_block import render_code_block
from app.ui.components.file_uploader import render_uploader
from app.ui.components.memory_view import render_memory_view

logger = logging.getLogger(__name__)


def _build_orchestrator(settings):
    """Wire all dependencies once per process."""
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
    from app.pipelines.explain_pipeline import ExplainPipeline
    from app.pipelines.generate_pipeline import GeneratePipeline
    from app.domain.enums import Intent
    from app.infrastructure.llm.freewhirr_client import FreeWhirrClient


    llm = FreeWhirrClient(settings)
    embedder = SentenceTransformerEmbedder(settings.embedder_model)
    vdb = ChromaVectorStore(settings.chroma_path, settings.chroma_collection)
    runner = SubprocessSandbox(settings.exec_timeout_s, settings.exec_mem_mb)
    memory_store = SQLiteMemoryStore(settings.sessions_db)
    chunker = CodeAwareChunker()

    memory = MemoryManager(memory_store, window=settings.memory_window)
    classifier = IntentClassifier(llm, settings.classifier_min_confidence)
    retriever = Retriever(embedder, vdb, top_k=settings.retrieval_top_k)
    evaluator = RelevanceEvaluator(llm, settings.relevance_threshold)
    generator = CodeGenerator(llm, memory_store)
    explainer = CodeExplainer(llm, memory_store)
    ingestion = IngestionService(embedder, vdb, chunker)

    router = (
        Router()
        .register(Intent.EXPLAIN_CODE, ExplainPipeline(explainer))
        .register(Intent.GENERATE_CODE, GeneratePipeline(retriever, evaluator, generator))
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
    return orchestrator, runner, memory, vdb


@st.cache_resource(show_spinner=False)
def get_runtime():
    settings = get_settings()
    return _build_orchestrator(settings)


def main() -> None:
    st.set_page_config(page_title="AI Coding Assistant", page_icon="🤖", layout="wide")
    inject_styles()
    init_state()
    settings = get_settings()
    st.session_state.max_file_mb = settings.max_file_mb


    orchestrator, runner, memory, vdb = get_runtime()

    # ── Sidebar ──────────────────────────────────
    with st.sidebar:
        st.title("🤖 AI Coding Assistant")
        st.caption("Free models · OpenRouter + local embeddings")
        if st.button("🆕 New session"):
            memory.clear(st.session_state.session_id)
            reset_session()
            st.rerun()
        st.divider()
        uploaded_text = render_uploader()
        st.divider()
        st.caption(f"Knowledge base: **{vdb.count()}** chunks")

    render_memory_view(memory)

    # ── Chat ─────────────────────────────────────
    render_history()

    user_input = st.chat_input("Ask me to explain code or generate code…")
    if not user_input and uploaded_text:
        user_input = f"Use this uploaded content as context:\n\n{uploaded_text[:5000]}"

    if user_input:
        append_message("user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            buffer: list[str] = []
            final_meta: dict = {}

            try:
                agen = orchestrator.handle(st.session_state.session_id, user_input)
                for event in iter_async(agen):
                    if event.kind == "token":
                        buffer.append(event.payload)
                        placeholder.markdown("".join(buffer))
                    elif event.kind == "message":
                        buffer.append(event.payload)
                        placeholder.markdown("".join(buffer))
                    elif event.kind == "meta":
                        final_meta.update(event.payload)
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
            _render_code_blocks_with_runners(full_reply, runner)

        append_message("assistant", full_reply, meta=final_meta)

    with st.sidebar:
        st.divider()
        st.caption(
            "Try: *“Write a Python function to merge two sorted lists”*  \n"
            "or:  *“Explain this code: ```python\\ndef f(x): return x*2\\n```”*"
        )


def _render_code_blocks_with_runners(markdown_text: str, runner) -> None:
    """Parse fenced code blocks and add Run buttons below each."""
    import re
    pattern = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
    for i, match in enumerate(pattern.finditer(markdown_text)):
        lang = (match.group(1) or "text").lower()
        code = match.group(2)
        runnable = lang in {"python", "javascript", "bash", "py", "js"}
        if lang == "py":
            lang = "python"
        if lang == "js":
            lang = "javascript"
        if runnable:
            # Only render the Run button + stdin, NOT the code again.
            render_code_block(code, lang, runner, runnable=runnable)


if __name__ == "__main__":
    main()