from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field
from app.domain.enums import Relevance, MessageRole

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

# ── RAG Code Models ──
@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: str
    created_at: datetime = field(default_factory=_utcnow)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce plain strings into MessageRole enums automatically."""
        if not isinstance(self.role, MessageRole):
            try:
                object.__setattr__(self, "role", MessageRole(str(self.role).lower()))
            except ValueError:
                raise ValueError(
                    f"Invalid role {self.role!r}. "
                    f"Expected one of: {[r.value for r in MessageRole]}"
                ) from None

@dataclass(frozen=True, slots=True)
class Classification:
    intent: str
    confidence: float
    language: str | None = None
    extracted_code: str | None = None
    reasoning: str = ""

@dataclass(frozen=True, slots=True)
class RetrievedDoc:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    embedding: list[float] | None = None

@dataclass(frozen=True, slots=True)
class WebSearchResult:
    title: str
    url: str
    content: str

@dataclass(frozen=True, slots=True)
class ExecutionResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    timed_out: bool = False

@dataclass(frozen=True, slots=True)
class Verdict:
    relevance: Relevance
    score: float
    reason: str = ""

@dataclass(frozen=True, slots=True)
class IngestResult:
    success: bool
    message: str
    chunks_ingested: int = 0

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
    def done(cls) -> "StreamEvent":
        return cls("done", None)

# ── Voice/SQL Models ──
class ColumnSchema(BaseModel):
    name: str
    type: str

# ── API Request/Response Models ──
class CodeGenerateRequest(BaseModel):
    session_id: str = Field(..., description="Unique chat session ID")
    query: str = Field(..., description="The user's prompt")

class CodeGenerateResponse(BaseModel):
    generated_text: str = Field(..., description="The Markdown response containing the code")

class CodeExplainRequest(BaseModel):
    session_id: str = Field(..., description="Unique chat session ID")
    code: str
    query: str = "Explain this code."

class CodeExplainResponse(BaseModel):
    explanation: str

class CodeExecuteRequest(BaseModel):
    code: str
    language: str = Field(..., description="python, javascript, bash, or sql")

class TranscriptionResponse(BaseModel):
    transcript: str
    detected_language: str

class DatasetUploadResponse(BaseModel):
    filename: str
    table_name: str
    row_count: int
    columns: list[ColumnSchema]

class AnalyzeRequest(BaseModel):
    dataset_name: str
    user_text: str

class AnalyzeResponse(BaseModel):
    user_text: str
    generated_sql: str
    result_data: list[dict[str, Any]]
    explanation: str