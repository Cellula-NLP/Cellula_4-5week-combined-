"""Minimal custom CSS for the chat UI."""
import streamlit as st

_CSS = """
<style>
.block-container { padding-top: 1.5rem; max-width: 1100px; }
.stChatMessage { border-radius: 10px; }
.code-header {
    font-size: 0.8rem; color: #888; margin-bottom: 0.2rem;
}
footer { visibility: hidden; }
</style>
"""


def inject_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)