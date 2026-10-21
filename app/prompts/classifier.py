"""Prompt for the intent classifier (Not used in FastAPI, but kept for compatibility)."""
from __future__ import annotations

_SYSTEM = """You are an intent classifier. Respond with STRICT JSON."""

def build_classifier_prompt(query: str, code_context: str | None, history: list[dict] | None = None) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": _SYSTEM}]
    if history:
        for m in history[-4:]:
            messages.append(m)

    user_block = f"Classify this request:\n{query}"
    if code_context:
        user_block += f"\n\nAttached code:\n```\n{code_context[:3000]}\n```"
    messages.append({"role": "user", "content": user_block})
    return messages