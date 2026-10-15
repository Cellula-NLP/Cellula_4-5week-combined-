"""Chat rendering helpers — user bubbles preserve whitespace exactly."""
from __future__ import annotations

import html

import streamlit as st

from app.ui.components.code_block import render_reply_controls


def render_history(executor=None) -> None:
    """Render all prior messages from session state.

    Each assistant message gets a deterministic ``message_id`` (its
    index in the list) so Run buttons keep stable keys across reruns.
    """
    for idx, msg in enumerate(st.session_state.get("messages", [])):
        if msg["role"] == "user":
            _render_user_bubble(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])
                if executor is not None:
                    render_reply_controls(
                        msg["content"], executor,
                        message_id=f"msg{idx}",
                    )


def append_message(role: str, content: str, meta: dict | None = None) -> None:
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "meta": meta or {},
    })


def render_user_bubble(content: str) -> None:
    _render_user_bubble(content)


def _render_user_bubble(content: str) -> None:
    safe = html.escape(content)
    st.markdown(
        f'<div class="user-row"><div class="user-bubble">{safe}</div></div>',
        unsafe_allow_html=True,
    )