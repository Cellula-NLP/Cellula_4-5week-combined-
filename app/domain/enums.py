"""Enumerations used across the application."""
from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    """Possible user intents after classification."""
    EXPLAIN_CODE = "explain_code"
    GENERATE_CODE = "generate_code"
    CLARIFY = "clarify"
    OUT_OF_SCOPE = "out_of_scope"


class Relevance(str, Enum):
    """Result of the relevance evaluation step."""
    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"
    UNKNOWN = "unknown"


class ConvState(str, Enum):
    """Conversation finite-state-machine states."""
    IDLE = "idle"
    AWAITING_CLARIFICATION = "awaiting_clarification"


class MessageRole(str, Enum):
    """OpenAI-compatible message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"