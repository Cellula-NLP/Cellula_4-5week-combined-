"""Render a code block with a Run button (sandboxed execution)."""
from __future__ import annotations

import uuid

import streamlit as st

from app.domain.models import ExecutionResult


def render_code_block(
    code: str,
    language: str,
    executor,
    *,
    runnable: bool = True,
) -> None:
    """Display code with syntax highlighting and an optional Run button."""
    st.code(code, language=language, line_numbers=True)

    if not runnable:
        return

    key = f"run_{uuid.uuid4().hex[:8]}"
    col_run, col_stdin = st.columns([1, 4])
    with col_run:
        run_clicked = st.button("▶ Run", key=key)
    with col_stdin:
        stdin = st.text_input(
            "stdin (optional)", key=f"stdin_{key}", label_visibility="collapsed",
            placeholder="Input passed to the program via stdin…",
        )

    if run_clicked:
        with st.spinner("Executing in sandbox…"):
            try:
                result: ExecutionResult = executor.execute(
                    code, language, stdin=stdin or "",
                )
            except Exception as exc:
                st.error(f"Execution error: {exc}")
                return
        _render_result(result)


def _render_result(result: ExecutionResult) -> None:
    st.session_state.last_execution = {
        "ok": result.ok, "exit_code": result.exit_code, "duration_ms": result.duration_ms,
    }
    if result.ok:
        st.success(f"✅ Exit {result.exit_code} · {result.duration_ms} ms")
    elif result.timed_out:
        st.error(f"⏱ Timed out after {result.duration_ms} ms")
    else:
        st.error(f"❌ Exit {result.exit_code} · {result.duration_ms} ms")

    if result.stdout:
        st.markdown("**stdout**")
        st.code(result.stdout, language="text")
    if result.stderr:
        st.markdown("**stderr**")
        st.code(result.stderr, language="text")
    if not result.stdout and not result.stderr:
        st.caption("(no output)")