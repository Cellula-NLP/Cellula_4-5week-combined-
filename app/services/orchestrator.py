"""Main coordinator: state machine + classification + routing."""
from __future__ import annotations

import logging
import re
from typing import AsyncIterator

from app.core.config import Settings
from app.domain.enums import ConvState, Intent
from app.domain.models import Classification, StreamEvent
from app.pipelines.base import PipelineContext
from app.services.intent_classifier import IntentClassifier
from app.services.memory_manager import MemoryManager
from app.services.router import Router

logger = logging.getLogger(__name__)


# ── Code extraction (module-level helpers) ────────────────────────

# Matches ```lang\n...\n``` and ```\n...\n``` (lang optional).
_CODE_FENCE = re.compile(r"```[a-zA-Z0-9_+\-]*\n(.*?)```", re.DOTALL)

# Matches `inline` code spans.
_INLINE_CODE = re.compile(r"`([^`\n]+)`")

# Prefixes that strongly suggest a line is code.
_CODE_LINE_PREFIXES: tuple[str, ...] = (
    "def ", "class ", "async def ", "import ", "from ",
    "return ", "if ", "elif ", "else:", "for ", "while ",
    "try:", "except", "finally:", "with ", "lambda ",
    "print(", "self.", "@", "yield ", "await ",
)

# Substrings that also suggest code (assignments, comparisons, calls).
_CODE_LINE_SUBSTRINGS: tuple[str, ...] = (
    " = ", ":=", "==", "!=", ">=", "<=", "->", "()", "):",
)


def _looks_like_code_line(line: str) -> bool:
    """True if the line looks like a code snippet rather than prose."""
    s = line.strip()
    if not s:
        return False
    if s.startswith(_CODE_LINE_PREFIXES):
        return True
    return any(tok in s for tok in _CODE_LINE_SUBSTRINGS)


def _looks_like_snippet(text: str) -> bool:
    """True if the whole text is short and contains at least one code line."""
    if len(text) > 800:
        return False
    lines = [line for line in text.split("\n") if line.strip()]
    return any(_looks_like_code_line(line) for line in lines)


def _extract_code(text: str) -> str | None:
    """Extract code from a user message.

    Order of preference:
      1. ```fenced``` blocks
      2. Text after the last ':' on a line if the tail looks like code
      3. Whole message if it looks like a snippet
      4. First `inline` code span
    """
    if not text or not text.strip():
        return None

    # 1) Fenced code block.
    match = _CODE_FENCE.search(text)
    if match:
        return match.group(1).strip()

    # 2) Everything after ':' if the tail looks like code.
    for line in text.split("\n"):
        if ":" in line:
            _, _, after = line.partition(":")
            after = after.strip()
            if after and _looks_like_code_line(after):
                idx = text.index(line)
                return text[idx:].split(":", 1)[1].strip()

    # 3) Whole-message snippet.
    if _looks_like_snippet(text):
        return text.strip()

    # 4) Inline code span.
    inline = _INLINE_CODE.search(text)
    if inline:
        return inline.group(1).strip()

    return None


def _looks_like_solution(text: str) -> bool:
    """Detect whether a user message is a solution rather than a new question."""
    tl = text.lower()
    if "```" in text and len(text) > 60:
        return True
    markers = (
        "here is the", "correct solution", "use this", "the answer is",
        "try this", "the implementation is", "here's the solution",
        "this should work", "you can use this",
    )
    return any(m in tl for m in markers)


class ConversationOrchestrator:
    """Stateless coordinator. Per-session state lives in MemoryStore."""

    # Recognized answers to "Explain or Generate?".
    _EXPLAIN_CHOICES: frozenset[str] = frozenset({
        "explain", "explain the code", "explain code", "explain it",
        "the first one", "1", "a",
    })
    _GENERATE_CHOICES: frozenset[str] = frozenset({
        "generate", "generate code", "generate new code", "generate it",
        "the second one", "2", "b",
    })

    def __init__(
        self,
        settings: Settings,
        classifier: IntentClassifier,
        router: Router,
        memory: MemoryManager,
        ingestion,
        state_store: dict[str, ConvState],
    ) -> None:
        self._settings = settings
        self._classifier = classifier
        self._router = router
        self._memory = memory
        self._ingestion = ingestion
        self._state_store = state_store  # in-memory FSM per session

    # ── Public entrypoint ─────────────────────────

    async def handle(
        self,
        session_id: str,
        user_input: str,
    ) -> AsyncIterator[StreamEvent]:
        # ── Guard: input validation ──────────────
        if not user_input or not user_input.strip():
            yield StreamEvent.error("Your message is empty. Please type something.")
            yield StreamEvent.done()
            return

        if len(user_input) > self._settings.max_input_chars:
            user_input = user_input[: self._settings.max_input_chars]
            logger.warning("Input truncated to %d chars.", self._settings.max_input_chars)

        # ── Recover from a pending clarification ─
        resolved = self._try_resolve_clarification(session_id, user_input)
        if resolved is not None:
            classification, recovered_code, recovered_query = resolved
            self._memory.add_user(session_id, user_input)
            logger.info(
                "Clarification resolved → %s (code=%s)",
                classification.intent,
                "yes" if recovered_code else "no",
            )
            async for event in self._dispatch(
                session_id=session_id,
                query=recovered_query,
                classification=classification,
                code=recovered_code,
            ):
                yield event
            return

        # ── State guard: awaiting solution from prior Not-Relevant ──
        if self._state_store.get(session_id) == ConvState.AWAITING_SOLUTION:
            if _looks_like_solution(user_input):
                pending_query = self._memory.profile(session_id).get("pending_query", "")
                result = self._ingestion.ingest_solution(
                    session_id=session_id,
                    solution_text=user_input,
                    original_query=pending_query,
                )
                self._state_store[session_id] = ConvState.IDLE
                self._memory.add_assistant(session_id, result.message)
                yield StreamEvent.message(result.message)
                yield StreamEvent.done()
                return
            # User asked something new → drop the AWAITING_SOLUTION state.
            self._state_store[session_id] = ConvState.IDLE

        # ── Persist user turn ────────────────────
        self._memory.add_user(session_id, user_input)

        # ── Extract code + classify ──────────────
        code = _extract_code(user_input)
        try:
            classification = await self._classifier.classify(user_input, code)
        except Exception as exc:
            logger.exception("Classification failed")
            yield StreamEvent.error(f"Could not classify your request: {exc}")
            yield StreamEvent.done()
            return

        logger.info(
            "Classified as %s (conf=%.2f)",
            classification.intent, classification.confidence,
        )

        # ── Handle CLARIFY ───────────────────────
        if classification.intent == Intent.CLARIFY:
            self._state_store[session_id] = ConvState.AWAITING_CLARIFICATION
            # Remember the pending code/query for the next turn.
            self._memory.update_preferences(
                session_id,
                pending_code=code or "",
                pending_query=user_input,
            )
            msg = (
                "I want to make sure I help correctly. Do you want me to:\n"
                "- **Explain** the code, or\n"
                "- **Generate** new code?"
            )
            self._memory.add_assistant(session_id, msg)
            yield StreamEvent.message(msg)
            yield StreamEvent.done()
            return

        # ── Handle OUT_OF_SCOPE ──────────────────
        if classification.intent == Intent.OUT_OF_SCOPE:
            msg = (
                "I'm specialized in programming tasks — code generation and "
                "explanation. Could you rephrase it as a coding question?"
            )
            self._memory.add_assistant(session_id, msg)
            yield StreamEvent.message(msg)
            yield StreamEvent.done()
            return

        # ── Dispatch to the matching pipeline ────
        async for event in self._dispatch(
            session_id=session_id,
            query=user_input,
            classification=classification,
            code=code,
        ):
            yield event

    # ── Dispatch ──────────────────────────────────

    async def _dispatch(
        self,
        *,
        session_id: str,
        query: str,
        classification: Classification,
        code: str | None,
    ) -> AsyncIterator[StreamEvent]:
        """Route the classified request to the appropriate pipeline."""
        try:
            pipeline = self._router.route(classification.intent)
        except ValueError as exc:
            logger.error("Routing failed: %s", exc)
            yield StreamEvent.error("Internal routing error. Please try again.")
            yield StreamEvent.done()
            return

        context = PipelineContext(
            code=code or classification.extracted_code,
            language=classification.language,
            classification=classification,
            top_k=self._settings.retrieval_top_k,
        )

        try:
            async for event in pipeline.run(
                session_id=session_id, query=query, context=context,
            ):
                # Keep the FSM in sync with pipeline-driven state changes.
                if event.kind == "state_change":
                    try:
                        self._state_store[session_id] = ConvState(event.payload)
                    except ValueError:
                        logger.warning(
                            "Unknown state_change payload: %s", event.payload,
                        )
                yield event
        except Exception as exc:
            logger.exception("Pipeline execution failed")
            yield StreamEvent.error(f"Something went wrong: {exc}")
            yield StreamEvent.done()

    # ── Clarification recovery ────────────────────

    def _try_resolve_clarification(
        self,
        session_id: str,
        user_input: str,
    ) -> tuple[Classification, str | None, str] | None:
        """If awaiting a clarify reply, interpret the user's answer.

        Returns ``(classification, recovered_code, recovered_query)`` on
        success, or ``None`` if we are not in clarification state or the
        answer isn't a recognized choice.
        """
        if self._state_store.get(session_id) != ConvState.AWAITING_CLARIFICATION:
            return None

        ans = user_input.strip().lower()
        profile = self._memory.profile(session_id)
        pending_code = (profile.get("pending_code") or "").strip() or None
        pending_query = (profile.get("pending_query") or "").strip()

        # The clarification turn is over either way — reset the FSM.
        self._state_store[session_id] = ConvState.IDLE
        # Clear pending fields so they don't leak into later turns.
        self._memory.update_preferences(
            session_id, pending_code=None, pending_query=None,
        )

        if ans in self._EXPLAIN_CHOICES:
            return (
                Classification(
                    intent=Intent.EXPLAIN_CODE,
                    confidence=1.0,
                    extracted_code=pending_code,
                    reasoning=pending_query or user_input,
                ),
                pending_code,
                pending_query or user_input,
            )

        if ans in self._GENERATE_CHOICES:
            return (
                Classification(
                    intent=Intent.GENERATE_CODE,
                    confidence=1.0,
                    extracted_code=None,
                    reasoning=pending_query or user_input,
                ),
                None,
                pending_query or user_input,
            )

        # Not a valid choice → fall through to normal classification.
        logger.info(
            "Clarification answer %r not recognized; re-classifying.", ans,
        )
        return None