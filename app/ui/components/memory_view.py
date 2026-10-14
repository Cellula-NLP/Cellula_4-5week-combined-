"""Optional memory-visualisation panel (sidebar expander)."""
from __future__ import annotations

import streamlit as st


def render_memory_view(memory) -> None:
    with st.sidebar.expander("🧠 Memory", expanded=False):
        msgs = st.session_state.get("messages", [])
        st.caption(f"Session ID: `{st.session_state.get('session_id', '—')}`")
        st.caption(f"Turns in memory: {len(msgs)}")
        st.caption(f"Conversation state: `{st.session_state.get('conv_state', 'idle')}`")

        last = st.session_state.get("last_execution")
        if last:
            status = "✅" if last["ok"] else "❌"
            st.caption(
                f"Last execution: {status} exit={last['exit_code']} "
                f"({last['duration_ms']} ms)"
            )

        profile = memory.profile(st.session_state.get("session_id", ""))
        if profile:
            st.json(profile)

        if st.button("🗑 Clear session"):
            from app.ui.state import reset_session
            reset_session()
            st.rerun()