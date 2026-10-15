"""File uploader rendered inside a popover (not the sidebar).

Attached files are wrapped in Markdown fenced blocks with a language
tag inferred from the extension, so the orchestrator's code extractor
picks them up naturally.

The uploader uses a *dynamic key* (bumped via ``st.session_state.upload_counter``)
so that after a file is consumed, the widget resets to a clean state and
does NOT re-attach the same file to the next message.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.core.constants import ALLOWED_UPLOAD_EXTENSIONS

# Cap so the composed message stays under max_input_chars (20,000).
_MAX_ATTACH_CHARS = 12_000

# Extension → Markdown fence language tag.
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "jsx", ".tsx": "tsx", ".java": "java", ".go": "go",
    ".rs": "rust", ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".kt": "kotlin",
    ".md": "markdown", ".txt": "text", ".pdf": "text",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".html": "html", ".css": "css", ".sh": "bash", ".sql": "sql",
}


def render_upload_popover() -> None:
    """Render the attach-file popover. Updates ``st.session_state.uploads``.

    Uses a dynamic widget key so the uploader resets after every consume.
    """
    with st.popover("📎", help="Attach a file"):
        st.markdown("**Attach a file**")
        st.caption(
            "Text, PDF, and code files are supported. "
            "The content will be included with your next message."
        )

        # ── Dynamic key = fresh widget after each consume ──
        counter = st.session_state.get("upload_counter", 0)
        uploaded = st.file_uploader(
            "Choose a file",
            type=[e.lstrip(".") for e in ALLOWED_UPLOAD_EXTENSIONS],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key=f"file_uploader_widget_{counter}",
        )

        if uploaded is None:
            _render_existing_uploads()
            return

        size_mb = uploaded.size / (1024 * 1024)
        if size_mb > 10:
            st.error(f"File too large ({size_mb:.1f} MB). Limit is 10 MB.")
            _render_existing_uploads()
            return

        text = _read_uploaded(uploaded)
        if text is None:
            _render_existing_uploads()
            return

        # Avoid re-storing the same file on every rerun while the popover is
        # open. Compare by name + size (cheap fingerprint).
        fingerprint = f"{uploaded.name}:{uploaded.size}"
        current = st.session_state.get("uploads") or []
        already_stored = any(
            u.get("fingerprint") == fingerprint for u in current
        )
        if not already_stored:
            st.session_state.uploads = [{
                "name": uploaded.name,
                "text": text,
                "fingerprint": fingerprint,
            }]

        st.success(f"✓ {uploaded.name} attached ({len(text):,} chars)")
        _render_existing_uploads()


def consume_uploads() -> str:
    """Pop pending uploads and return them as fenced Markdown blocks.

    Bumps ``upload_counter`` so the file uploader resets to a clean state
    on the next render — this is what prevents the same file from being
    re-attached to subsequent messages.
    """
    uploads = st.session_state.get("uploads") or []
    if not uploads:
        return ""

    # Clear pending uploads + reset the uploader widget.
    st.session_state.uploads = []
    st.session_state.upload_counter = st.session_state.get("upload_counter", 0) + 1

    blocks: list[str] = []
    for u in uploads:
        suffix = Path(u["name"]).suffix.lower()
        lang = _EXT_TO_LANG.get(suffix, "")
        text = u["text"]

        if len(text) > _MAX_ATTACH_CHARS:
            text = text[:_MAX_ATTACH_CHARS] + "\n# …[truncated]…"

        # Keep the outer fence intact even if the file has its own fences.
        if "```" in text:
            text = text.replace("```", "'''")

        blocks.append(f"```{lang}\n# File: {u['name']}\n{text}\n```")

    return "\n\n".join(blocks)


# ── Internals ─────────────────────────────────────────

def _render_existing_uploads() -> None:
    uploads = st.session_state.get("uploads") or []
    if not uploads:
        return
    st.divider()
    st.caption("Pending attachments")
    for u in uploads:
        st.markdown(f"- 📄 `{u['name']}`")


def _read_uploaded(uploaded) -> str | None:
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