"""Chat rendering helpers."""
from __future__ import annotations

import streamlit as st


def render_history() -> None:
    """Render all prior messages from session state."""
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)


def append_message(role: str, content: str, meta: dict | None = None) -> None:
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "meta": meta or {},
    })