"""Prompt for the intent classifier."""
from __future__ import annotations

from app.domain.models import Message

_SYSTEM = """You are an intent classifier for a coding assistant.

Classify the user's request into EXACTLY ONE of these intents:
- "explain_code": user wants an explanation of existing code.
- "generate_code": user wants new code generated.
- "clarify": the request is ambiguous and you cannot confidently decide.
- "out_of_scope": the request is not about programming/code at all.

Respond with STRICT JSON only (no markdown, no prose):
{
  "intent": "explain_code" | "generate_code" | "clarify" | "out_of_scope",
  "confidence": 0.0-1.0,
  "language": "python" | "javascript" | ... | null,
  "reasoning": "one short sentence"
}

Rules:
- If the user pastes code and asks what it does, choose "explain_code".
- If the user asks to write/build/create code, choose "generate_code".
- If both, prefer "explain_code" unless the request is mostly to produce new code.
- If confidence < 0.55, return "clarify".
"""


def build_classifier_prompt(query: str, code_context: str | None) -> list[Message]:
    user_block = f"User request:\n{query}"
    if code_context:
        user_block += f"\n\nAttached code:\n```\n{code_context[:3000]}\n```"
    return [
        Message(role="system", content=_SYSTEM),
        Message(role="user", content=user_block),
    ]