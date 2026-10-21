from __future__ import annotations
from enum import Enum

class Intent(str, Enum):
    EXPLAIN_CODE = "explain_code"
    GENERATE_CODE = "generate_code"
    CLARIFY = "clarify"
    OUT_OF_SCOPE = "out_of_scope"

class Relevance(str, Enum):
    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"
    UNKNOWN = "unknown"

class ConvState(str, Enum):
    IDLE = "idle"
    AWAITING_CLARIFICATION = "awaiting_clarification"

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"