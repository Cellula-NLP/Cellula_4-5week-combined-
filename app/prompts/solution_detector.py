"""Prompt for detecting whether a user message is a solution (used by FSM)."""
from __future__ import annotations

from app.domain.models import Message

_SYSTEM = """You decide if a user's message is providing a SOLUTION to a previous coding question.

Respond with STRICT JSON only:
{
  "is_solution": true | false,
  "confidence": 0.0-1.0
}

A solution contains code, a direct answer, or an implementation.
A non-solution is a new question, a greeting, or unrelated chatter.
"""


def build_solution_detector_prompt(user_message: str) -> list[Message]:
    return [
        Message(role="system", content=_SYSTEM),
        Message(role="user", content=user_message[:2000]),
    ]