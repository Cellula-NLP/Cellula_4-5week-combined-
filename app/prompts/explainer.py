"""Prompt for code explanation (NO RAG)."""
from __future__ import annotations
from app.domain.enums import MessageRole

from app.domain.models import Message

_SYSTEM = """You are a senior software engineer who explains code clearly.

Rules:
- Explain WHAT the code does, HOW it works, and WHY.
- Walk through it line-by-line or block-by-block as appropriate.
- Point out edge cases, bugs, or anti-patterns if present.
- Use concise Markdown (headings, bullet points, inline code).
- If the user's language preference is known, answer in that language.
- Do NOT invent code that isn't in the user's snippet.
"""


def build_explainer_prompt(
    query: str,
    code: str | None,
    history: list[Message],
    language: str | None = None,
) -> list[Message]:
    messages: list[Message] = [Message(role="system", content=_SYSTEM)]

    # Prepend a short history window (last few turns) for context.
    for m in history[-6:]:
        if m.role in (MessageRole.USER, MessageRole.ASSISTANT):
            messages.append(m)

    lang_hint = f"\n(Preferred language: {language})" if language else ""
    body = f"{query}{lang_hint}"
    if code:
        body += f"\n\nCode to explain:\n```\n{code}\n```"
    messages.append(Message(role="user", content=body))
    return messages