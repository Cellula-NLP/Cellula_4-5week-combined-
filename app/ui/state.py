"""Session-state schema and helpers."""
from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

_DEFAULTS: dict[str, Any] = {
    "session_id": None,
    "messages": [],          # list of {"role": str, "content": str, "meta": dict}
    "conv_state": "idle",
    "pending_query": None,
    "settings": {"model": "qwen/qwen3-coder:free", "top_k": 4},
    "last_execution": None,
    "uploads": [],
}


def init_state() -> None:
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.session_id is None:
        st.session_state.session_id = str(uuid.uuid4())


def reset_session() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.conv_state = "idle"
    st.session_state.pending_query = None
    st.session_state.last_execution = None
    st.session_state.uploads = []