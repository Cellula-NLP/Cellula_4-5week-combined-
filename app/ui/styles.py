"""Global CSS — teal-themed, Claude-like UI."""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* ── Theme tokens ─────────────────────────────────────── */
:root {
    --teal-50:  #f0fdfa;
    --teal-100: #ccfbf1;
    --teal-200: #99f6e4;
    --teal-300: #5eead4;
    --teal-400: #2dd4bf;
    --teal-500: #14b8a6;
    --teal-600: #0d9488;
    --teal-700: #0f766e;

    --bg-deep:     #0a1620;
    --bg-surface:  #0f2231;
    --bg-elevated: #142d40;
    --bg-hover:    #17384d;

    --text-primary: #e0f2f1;
    --text-muted:   #7d99a8;
    --text-soft:    #b3ccd1;

    --border-subtle: rgba(45, 212, 191, 0.14);
    --border-strong: rgba(45, 212, 191, 0.35);
    --glow:          rgba(45, 212, 191, 0.18);

    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 18px;
    --radius-pill: 999px;

    --ease: cubic-bezier(0.16, 1, 0.3, 1);
}

/* ── Global layout ────────────────────────────────────── */
.stApp {
    background:
        radial-gradient(1200px 600px at 15% -10%, rgba(45, 212, 191, 0.08), transparent 60%),
        radial-gradient(900px 500px at 95% 110%, rgba(20, 184, 166, 0.06), transparent 60%),
        var(--bg-deep);
    color: var(--text-primary);
}

header[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: none !important;
    height: 0 !important;
    min-height: 0 !important;
}

.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 4rem !important;
    max-width: 980px;
}

/* ── Typography ───────────────────────────────────────── */
h1, h2, h3, h4 { color: var(--text-primary) !important; letter-spacing: -0.01em; }
h1 { font-weight: 700; }
h2 { font-weight: 650; }
p, li, span, label { color: var(--text-primary); }
code {
    background: rgba(45, 212, 191, 0.08) !important;
    color: var(--teal-300) !important;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.9em;
}

/* ── Sidebar (unused) ─────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1c2a 0%, #0a1620 100%);
    border-right: 1px solid var(--border-subtle);
}

/* ── Header bar ───────────────────────────────────────── */
.app-header-left {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding-top: 4px;
}
.app-logo {
    width: 42px; height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--teal-400), var(--teal-600));
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow:
        0 6px 20px rgba(45, 212, 191, 0.28),
        inset 0 1px 0 rgba(255, 255, 255, 0.22);
}
.app-title {
    font-size: 1.2rem;
    font-weight: 650;
    color: var(--text-primary);
    line-height: 1.15;
}
.app-subtitle {
    font-size: 0.76rem;
    color: var(--text-muted);
    margin-top: 2px;
}
.header-rule {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-subtle) 20%, var(--border-subtle) 80%, transparent);
    margin: 0.9rem 0 1.4rem 0;
}

/* ── User message bubble (preserves whitespace) ───────── */
.user-row {
    display: flex;
    justify-content: flex-end;
    margin: 0.6rem 0;
    animation: fadeUp 0.35s var(--ease);
}
.user-bubble {
    max-width: 82%;
    padding: 0.85rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    background: linear-gradient(135deg,
        rgba(45, 212, 191, 0.16) 0%,
        rgba(13, 148, 136, 0.14) 100%);
    border: 1px solid rgba(45, 212, 191, 0.28);
    color: var(--text-primary);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    word-break: break-word;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 0.95rem;
    line-height: 1.55;
    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.02) inset;
}

/* ── Assistant message ───────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.4rem 0 !important;
    animation: fadeUp 0.4s var(--ease);
}
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, var(--teal-400), var(--teal-600)) !important;
    color: #062024 !important;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #12304a, #0d2235) !important;
    border: 1px solid var(--border-subtle);
}

/* ── Thinking indicator ──────────────────────────────── */
.thinking-box {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    padding: 0.6rem 0;
    color: var(--text-soft);
    font-size: 0.9rem;
    animation: fadeUp 0.3s var(--ease);
}
.thinking-dots {
    display: inline-flex;
    gap: 5px;
}
.thinking-dots span {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--teal-400);
    box-shadow: 0 0 10px var(--glow);
    animation: dotPulse 1.4s ease-in-out infinite;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
.thinking-label {
    color: var(--teal-300);
    font-weight: 500;
    letter-spacing: -0.005em;
}

@keyframes dotPulse {
    0%, 60%, 100% { opacity: 0.35; transform: scale(0.8); }
    30%           { opacity: 1;    transform: scale(1.05); }
}

/* ── Streaming cursor ────────────────────────────────── */
.streaming-cursor {
    display: inline-block;
    width: 8px;
    height: 1.05em;
    background: var(--teal-400);
    margin-left: 3px;
    vertical-align: text-bottom;
    border-radius: 2px;
    box-shadow: 0 0 8px var(--glow);
    animation: cursorBlink 1s steps(2) infinite;
}
@keyframes cursorBlink {
    0%, 50%   { opacity: 1; }
    51%, 100% { opacity: 0; }
}

/* ── Fade-in used by several elements ────────────────── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Code blocks ─────────────────────────────────────── */
.stCodeBlock, pre {
    background: #0a1c2a !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}
.stCodeBlock code {
    color: var(--text-primary) !important;
    background: transparent !important;
    font-size: 0.86rem !important;
    line-height: 1.55 !important;
}

/* ── Text area (multiline input) ─────────────────────── */
[data-testid="stTextArea"] textarea {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    color: var(--text-primary) !important;
    padding: 0.95rem 1.1rem !important;
    font-size: 0.95rem !important;
    line-height: 1.55 !important;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif !important;
    resize: vertical !important;
    transition: border-color 0.2s var(--ease), box-shadow 0.2s var(--ease);
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--teal-500) !important;
    box-shadow: 0 0 0 3px var(--glow) !important;
    outline: none !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: var(--text-muted) !important; }

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button {
    background: var(--bg-elevated);
    color: var(--text-primary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    font-weight: 500;
    transition: all 0.18s var(--ease);
}
.stButton > button:hover {
    background: var(--bg-hover);
    border-color: var(--border-strong);
    color: var(--teal-200);
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--teal-500), var(--teal-600));
    border: 1px solid var(--teal-500);
    color: #04222a;
    font-weight: 650;
    box-shadow: 0 4px 14px rgba(45, 212, 191, 0.25);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--teal-400), var(--teal-500));
    color: #04222a;
    box-shadow: 0 6px 18px rgba(45, 212, 191, 0.35);
    transform: translateY(-1px);
}

/* ── Popovers ────────────────────────────────────────── */
[data-testid="stPopover"] > button {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    color: var(--text-soft);
    font-weight: 500;
    height: 42px;
    min-width: 42px;
    padding: 0 !important;
    font-size: 1.1rem;
    transition: all 0.18s var(--ease);
}
[data-testid="stPopover"] > button:hover {
    border-color: var(--teal-500);
    background: var(--bg-hover);
    color: var(--teal-300);
}
[data-testid="stPopoverBody"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5) !important;
    padding: 1rem 1.1rem !important;
    min-width: 340px;
    max-width: 380px;
}

/* ── Metrics inside popover ──────────────────────────── */
[data-testid="stPopoverBody"] [data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
    color: var(--teal-300) !important;
    font-weight: 650;
}
[data-testid="stPopoverBody"] [data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── File uploader (inside popover) ──────────────────── */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(45, 212, 191, 0.04) !important;
    border: 1px dashed var(--border-strong) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(45, 212, 191, 0.08) !important;
}

/* ── Alerts ──────────────────────────────────────────── */
.stAlert {
    border-radius: var(--radius-md);
    border: 1px solid var(--border-subtle);
    background: var(--bg-surface);
}

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(45, 212, 191, 0.18);
    border-radius: 6px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(45, 212, 191, 0.3); }

/* ── Hide Streamlit chrome ───────────────────────────── */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
"""


def inject_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)