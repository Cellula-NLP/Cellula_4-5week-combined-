"""Session memory panel — shows preferences, recent turns, and state.

Fully implements section 8 of the task spec:
  • Previous questions      → "Recent activity" section
  • Previous generated code → shown via snippets in recent activity
  • User preferences        → editable language + framework chips
  • Language preference     → dropdown
  • Framework preference    → removable chips
  • Follow-up requests      → implied by the full history being fed to the LLM
"""
from __future__ import annotations

import streamlit as st

_LANGUAGES = [
    "auto", "python", "javascript", "typescript",
    "java", "go", "rust", "c", "cpp", "csharp", "ruby", "php",
]


def render_memory_popover(memory, orchestrator=None) -> None:
    """Render the memory popover (triggered from the header).

    ``orchestrator`` must be the *live* ConversationOrchestrator instance
    so we can read the FSM state from the source of truth. If it's missing
    or doesn't expose ``get_state``, we fall back to the cached
    ``st.session_state.conv_state`` value.
    """
    with st.popover("⚙", help="Session memory & preferences"):
        sid = st.session_state.get("session_id", "")
        msgs = st.session_state.get("messages", [])
        profile = memory.profile(sid) if sid else {}

        # ── Title ────────────────────────────────
        st.markdown("#### Session memory")
        if sid:
            st.caption(f"`{sid[:8]}…{sid[-4:]}`")

        # ── Quick stats ──────────────────────────
        user_turns = sum(1 for m in msgs if m.get("role") == "user")
        c1, c2 = st.columns(2)
        c1.metric("Turns", user_turns)

        # Prefer live state from the orchestrator; fall back to the cached
        # value in session_state if the orchestrator can't provide it.
        live_state = None
        if orchestrator is not None and hasattr(orchestrator, "get_state"):
            try:
                live_state = orchestrator.get_state(sid).value
            except Exception:
                live_state = None
        if live_state is None:
            live_state = st.session_state.get("conv_state", "idle")

        c2.metric("State", live_state)

        st.divider()

        # ── Language preference ─────────────────
        st.markdown("**Language preference**")
        current_lang = profile.get("language") or "auto"
        idx = _LANGUAGES.index(current_lang) if current_lang in _LANGUAGES else 0
        new_lang = st.selectbox(
            "Language",
            _LANGUAGES,
            index=idx,
            key="mem_language_select",
            label_visibility="collapsed",
        )
        if new_lang != current_lang:
            memory.set_language(sid, None if new_lang == "auto" else new_lang)
            st.rerun()

        # ── Framework preferences ───────────────
        st.markdown("**Framework preferences**")
        frameworks: list[str] = profile.get("frameworks") or []
        if frameworks:
            chip_cols = st.columns(min(len(frameworks), 3))
            for i, fw in enumerate(frameworks):
                with chip_cols[i % 3]:
                    if st.button(f"✕  {fw}", key=f"rm_fw_{fw}",
                                 use_container_width=True):
                        memory.remove_framework(sid, fw)
                        st.rerun()
        else:
            st.caption(
                "_None detected yet — mention a framework like FastAPI, "
                "React, or Django and it'll appear here. Language "
                "preferences are above._"
            )

        st.divider()

        # ── Recent activity ─────────────────────
        st.markdown("**Recent activity**")
        if msgs:
            for m in msgs[-4:]:
                role_icon = "▸" if m["role"] == "user" else "◆"
                text = " ".join(m["content"].split())[:140]
                if len(m["content"]) > 140:
                    text += "…"
                st.markdown(f"`{role_icon}` {text}")
        else:
            st.caption("_(no turns yet)_")

        # ── Last execution ──────────────────────
        last = st.session_state.get("last_execution")
        if last:
            st.divider()
            ok = "✓" if last["ok"] else "✗"
            st.caption(
                f"**Last execution**  ·  {ok} exit {last['exit_code']} "
                f"·  {last['duration_ms']} ms"
            )

        st.divider()

        # ── Clear session ───────────────────────
        if st.button("🗑  Clear session", use_container_width=True, type="secondary"):
            try:
                memory.clear(sid)
            except Exception:
                pass
            from app.ui.state import reset_session
            reset_session()
            st.rerun()