"""Prompt for code explanation (NO RAG)."""
from __future__ import annotations

_SYSTEM = """You are a senior software engineer who explains code clearly.
Rules:
- Explain WHAT the code does, HOW it works, and WHY.
- Point out edge cases or bugs if present.
- Use concise Markdown.
"""

def build_explainer_prompt(
    query: str,
    code: str | None,
    history: list[dict],
    language: str | None = None,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": _SYSTEM}]

    for m in history[-6:]:
        messages.append(m)

    body = f"{query}"
    if code:
        body += f"\n\nCode to explain:\n```\n{code}\n```"
    messages.append({"role": "user", "content": body})
    return messages