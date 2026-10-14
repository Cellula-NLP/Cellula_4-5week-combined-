"""Immutable domain models used by services and pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.domain.enums import Intent, MessageRole, Relevance


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Conversation ──────────────────────────────────

@dataclass(frozen=True, slots=True)
class Message:
    """A single conversation turn.

    Accepts `role` as either a MessageRole enum or a plain string
    ("user", "assistant", ...) and coerces it to the enum.
    """
    role: MessageRole
    content: str
    created_at: datetime = field(default_factory=_utcnow)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            try:
                object.__setattr__(self, "role", MessageRole(str(self.role).lower()))
            except ValueError:
                raise ValueError(
                    f"Invalid role {self.role!r}. "
                    f"Expected one of: {[r.value for r in MessageRole]}"
                ) from None


# ── Classification ────────────────────────────────

@dataclass(frozen=True, slots=True)
class Classification:
    intent: Intent
    confidence: float
    language: str | None = None
    extracted_code: str | None = None
    reasoning: str = ""


# ── Retrieval ─────────────────────────────────────

@dataclass(frozen=True, slots=True)
class RetrievedDoc:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    embedding: list[float] | None = None


@dataclass(frozen=True, slots=True)
class Verdict:
    relevance: Relevance
    score: float
    reason: str = ""


# ── Execution ─────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ExecutionResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    timed_out: bool = False


# ── Ingestion ─────────────────────────────────────

@dataclass(frozen=True, slots=True)
class IngestResult:
    success: bool
    message: str
    chunks_ingested: int = 0


# ── Streaming ─────────────────────────────────────

@dataclass(frozen=True, slots=True)
class StreamEvent:
    kind: str
    payload: Any = None

    @classmethod
    def token(cls, text: str) -> "StreamEvent":
        return cls("token", text)

    @classmethod
    def message(cls, text: str) -> "StreamEvent":
        return cls("message", text)

    @classmethod
    def meta(cls, data: dict) -> "StreamEvent":
        return cls("meta", data)

    @classmethod
    def state_change(cls, state: str) -> "StreamEvent":
        return cls("state_change", state)

    @classmethod
    def error(cls, text: str) -> "StreamEvent":
        return cls("error", text)

    @classmethod
    def done(cls) -> "StreamEvent":
        return cls("done", None)