"""Run button + reply-code-parsing utilities with persistent results.

Run buttons appear ONLY for responses that contain *generated* code
(the Generate route). Explanation responses contain illustrative
snippets that are meant to be read, not executed — showing Run
buttons under them is misleading (many are fragments that produce
no output or reference undefined names).

Additionally, Run buttons are hidden for any language the sandbox
cannot actually execute (e.g. JavaScript when ``node`` isn't
installed). The set is read from ``executor.available_languages``.

Key implementation details:
  • Button keys are deterministic (based on snippet_id), so Streamlit
    recognizes clicks across reruns.
  • Execution results are stored in st.session_state keyed by
    snippet_id, so they survive reruns.
"""
from __future__ import annotations

import re

import streamlit as st

from app.domain.models import ExecutionResult

_CODE_FENCE = re.compile(r"```([a-zA-Z0-9_+\-]*)\n(.*?)```", re.DOTALL)

# Canonical names the UI knows how to render a Run button for. The actual
# gate is executor.available_languages — this set is only the fallback
# when the executor can't tell us (degraded mode / older interface).
_RUNNABLE_FALLBACK = frozenset({"python", "javascript", "bash"})


# ── Reply parsing ──────────────────────────────────

def render_reply_controls(
    markdown_text: str,
    executor,
    *,
    message_id: str,
    show_run: bool = False,
) -> None:
    """Parse fenced code blocks in an assistant reply and add Run buttons.

    ``message_id``   — unique per assistant message (stable button keys).
    ``show_run``     — if False, no Run buttons are rendered at all.
                       Set to True only for generated code responses.
    """
    if not show_run or not markdown_text or executor is None:
        return

    # Ask the executor which languages it can actually run. If it can't
    # tell us, fall back to the conservative default set.
    available = getattr(executor, "available_languages", None) or _RUNNABLE_FALLBACK

    matches = list(_CODE_FENCE.finditer(markdown_text))
    for i, match in enumerate(matches):
        lang_raw = (match.group(1) or "").strip().lower()
        code = match.group(2)
        lang = _normalize_language(_infer_language(lang_raw, code))
        if lang not in available:
            continue

        snippet_id = f"{message_id}:snippet{i}"
        st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
        render_run_controls(
            code, lang, executor,
            runnable=True,
            label=f"Run snippet #{i + 1} · {lang}",
            snippet_id=snippet_id,
        )


# ── Run controls ───────────────────────────────────

def render_run_controls(
    code: str,
    language: str,
    executor,
    *,
    runnable: bool = True,
    label: str | None = None,
    snippet_id: str,
) -> None:
    """Render a Run button with a STABLE key. Executes the code as-is."""
    if not runnable:
        return

    if label:
        st.caption(label)

    key = f"run_{snippet_id}"
    result_key = f"result_{snippet_id}"

    col_run, col_info = st.columns([1, 4], gap="small")
    with col_run:
        run_clicked = st.button(
            "▶ Run", key=key, use_container_width=True,
        )
    with col_info:
        st.caption("Executes the code exactly as shown above.")

    stored: ExecutionResult | None = st.session_state.get(result_key)

    if run_clicked:
        with st.spinner("Executing in sandbox…"):
            try:
                stored = executor.execute(code, language)
            except Exception as exc:
                st.error(f"Execution error: {exc}")
                stored = None
            if stored is not None:
                st.session_state[result_key] = stored
                st.session_state.last_execution = {
                    "ok": stored.ok,
                    "exit_code": stored.exit_code,
                    "duration_ms": stored.duration_ms,
                }

    if stored is not None:
        _render_result(stored)


# ── Internals ──────────────────────────────────────

def _normalize_language(lang: str) -> str:
    return {"py": "python", "js": "javascript", "sh": "bash"}.get(lang, lang)


def _infer_language(lang: str, code: str) -> str:
    """Best-effort language detection when the fence tag is missing."""
    if lang:
        return lang
    if re.search(r"^\s*(def |class |import |from |async def )", code, re.M):
        return "python"
    if re.search(r"\b(print|self|None|elif|lambda)\b", code):
        return "python"
    if re.search(r"\b(function|const|let|=>|console\.log|require)\b", code):
        return "javascript"
    if re.search(r"^\s*(#!|echo |\$ )", code, re.M):
        return "bash"
    return ""


def _render_result(result: ExecutionResult) -> None:
    if result.ok:
        st.success(f"✓ Exit {result.exit_code} · {result.duration_ms} ms")
    elif result.timed_out:
        st.error(f"⏱ Timed out after {result.duration_ms} ms")
    else:
        st.error(f"✗ Exit {result.exit_code} · {result.duration_ms} ms")

    if result.stdout:
        st.markdown("**stdout**")
        st.code(result.stdout, language="text")
    if result.stderr:
        st.markdown("**stderr**")
        st.code(result.stderr, language="text")
    if not result.stdout and not result.stderr:
        st.caption("(no output)")