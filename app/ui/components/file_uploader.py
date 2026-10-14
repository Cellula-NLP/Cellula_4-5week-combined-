"""Sidebar file-uploader. Extracts text and stores it for the next turn."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.core.constants import ALLOWED_UPLOAD_EXTENSIONS


def render_uploader() -> str | None:
    """Render the uploader. Returns extracted text (or None)."""
    uploaded = st.file_uploader(
        "Upload a file (optional)",
        type=[e.lstrip(".") for e in ALLOWED_UPLOAD_EXTENSIONS],
        accept_multiple_files=False,
    )
    if uploaded is None:
        return None

    size_mb = uploaded.size / (1024 * 1024)
    if size_mb > st.session_state.get("max_file_mb", 10):
        st.error(f"File too large ({size_mb:.1f} MB). Limit is 10 MB.")
        return None

    suffix = Path(uploaded.name).suffix.lower()
    try:
        if suffix == ".pdf":
            return _read_pdf(uploaded)
        return uploaded.read().decode("utf-8", errors="replace")
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return None


def _read_pdf(file) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf is not installed. Run: pip install pypdf")
    reader = PdfReader(file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()