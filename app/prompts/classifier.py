"""Prompt for the intent classifier."""
from __future__ import annotations

from app.domain.enums import MessageRole
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
- If the message is a short follow-up (e.g. "add error handling",
  "make it faster", "convert to JavaScript"), infer intent from the
  conversation history — a follow-up asking to modify code is
  "generate_code".
- If both explanations and generation are requested, prefer
  "explain_code" unless the request is mostly to produce new code.
- If confidence < 0.55, return "clarify".
"""


def build_classifier_prompt(
    query: str,
    code_context: str | None,
    history: list[Message] | None = None,
) -> list[Message]:
    """Build the classifier prompt with optional conversation context."""
    messages: list[Message] = [Message(role=MessageRole.SYSTEM, content=_SYSTEM)]

    # Include the last few turns as context so follow-ups classify correctly.
    if history:
        for m in history[-4:]:
            if m.role in (MessageRole.USER, MessageRole.ASSISTANT):
                # Truncate assistant replies — we only need their gist.
                content = m.content[:400]
                messages.append(Message(role=m.role, content=content))

    user_block = f"Classify this request:\n{query}"
    if code_context:
        user_block += f"\n\nAttached code:\n```\n{code_context[:3000]}\n```"
    messages.append(Message(role=MessageRole.USER, content=user_block))
    return messages