"""Main coordinator: state machine + classification + routing."""
from __future__ import annotations

import logging
import re
from typing import AsyncIterator

from app.core.config import Settings
from app.domain.enums import ConvState, Intent, MessageRole
from app.domain.models import Classification, StreamEvent
from app.pipelines.base import PipelineContext
from app.services.framework_detector import detect_frameworks
from app.services.intent_classifier import IntentClassifier
from app.services.memory_manager import MemoryManager
from app.services.router import Router

logger = logging.getLogger(__name__)


# ── Code extraction (module-level helpers) ────────────────────────

_CODE_FENCE = re.compile(r"```[a-zA-Z0-9_+\-]*\n(.*?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")

_CODE_LINE_PREFIXES: tuple[str, ...] = (
    "def ", "class ", "async def ", "import ", "from ",
    "return ", "if ", "elif ", "else:", "for ", "while ",
    "try:", "except", "finally:", "with ", "lambda ",
    "print(", "self.", "@", "yield ", "await ",
)

_CODE_LINE_SUBSTRINGS: tuple[str, ...] = (
    " = ", ":=", "==", "!=", ">=", "<=", "->", "()", "):",
)


def _looks_like_code_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith(_CODE_LINE_PREFIXES):
        return True
    return any(tok in s for tok in _CODE_LINE_SUBSTRINGS)


def _looks_like_snippet(text: str) -> bool:
    if len(text) > 800:
        return False
    lines = [line for line in text.split("\n") if line.strip()]
    return any(_looks_like_code_line(line) for line in lines)


def _extract_code(text: str) -> str | None:
    """Extract code from a user message."""
    if not text or not text.strip():
        return None

    match = _CODE_FENCE.search(text)
    if match:
        return match.group(1).strip()

    for line in text.split("\n"):
        if ":" in line:
            _, _, after = line.partition(":")
            after = after.strip()
            if after and _looks_like_code_line(after):
                idx = text.index(line)
                return text[idx:].split(":", 1)[1].strip()

    if _looks_like_snippet(text):
        return text.strip()

    inline = _INLINE_CODE.search(text)
    if inline:
        return inline.group(1).strip()

    return None


def _extract_code_for_ingestion(text: str) -> str:
    """Pull code out of a user-provided solution."""
    text = (text or "").strip()
    if not text:
        return text

    match = _CODE_FENCE.search(text)
    if match:
        return match.group(1).strip()

    lines = text.split("\n")
    for idx, line in enumerate(lines):
        if _looks_like_code_line(line):
            return "\n".join(lines[idx:]).strip()

    return text


def _strip_fences(text: str) -> str:
    return _CODE_FENCE.sub("", text).strip()


# ── Solution detection ──────────────────────────────────────────

_QUESTION_MARKERS: tuple[str, ...] = (
    "explain", "what does", "what is", "what are", "what's", "whats",
    "why is", "why does", "why do", "why my",
    "how does", "how do", "how can", "how to",
    "can u", "can you", "could you", "could u",
    "tell me", "describe", "walk me", "help me", "show me",
    "?", "what u", "which ",
)


def _strip_attached_files(text: str) -> str:
    return _CODE_FENCE.sub("", text).strip()


def _has_question_marker(text: str) -> bool:
    tl = text.lower()
    return any(m in tl for m in _QUESTION_MARKERS)


def _looks_like_solution(text: str) -> bool:
    """Detect whether a user message is a *solution* rather than a question."""
    if not text or not text.strip():
        return False

    user_words = _strip_attached_files(text)
    if not user_words:
        return False

    if _has_question_marker(user_words):
        return False

    if _looks_like_snippet(user_words):
        return True

    if "```" in text and len(text) > 60 and len(user_words) < 30:
        return True

    tl = user_words.lower()
    markers = (
        "here is the", "correct solution", "use this", "the answer is",
        "try this", "the implementation is", "here's the solution",
        "this should work", "you can use this",
        "correct implementation", "reference implementation",
    )
    return any(m in tl for m in markers)


# ── Small-talk detection ────────────────────────────────────────

_SMALLTALK_GREETINGS = frozenset({
    "hi", "hello", "hey", "yo", "hi there", "hello there",
    "good morning", "good afternoon", "good evening", "howdy",
})

_SMALLTALK_THANKS = frozenset({
    "thanks", "thank you", "thx", "ty", "appreciate it",
    "thanks!", "thank you!", "cheers",
})

_SMALLTALK_GOODBYES = frozenset({
    "bye", "goodbye", "good bye", "see you", "later", "cya",
    "see ya", "talk later",
})


def _handle_smalltalk(text: str) -> str | None:
    if not text or not text.strip():
        return None

    tl = text.lower().strip().strip("!.,?!").strip()
    words = set(tl.split())

    if tl in _SMALLTALK_GREETINGS:
        return (
            "Hi! I can **explain code** or **generate code** for you.\n\n"
            "Try something like *\"merge two sorted lists\"*, or paste a "
            "snippet and ask me to explain it."
        )

    if tl in _SMALLTALK_THANKS or (words & _SMALLTALK_THANKS):
        return "You're welcome! Let me know if you need anything else."

    if tl in _SMALLTALK_GOODBYES or (words & _SMALLTALK_GOODBYES):
        return "Goodbye! Come back whenever you need help with code."

    return None


class ConversationOrchestrator:
    """Stateless coordinator. Per-session state lives in MemoryStore."""

    _EXPLAIN_CHOICES: frozenset[str] = frozenset({
        "explain", "explain the code", "explain code", "explain it",
        "the first one", "1", "a",
    })
    _GENERATE_CHOICES: frozenset[str] = frozenset({
        "generate", "generate code", "generate new code", "generate it",
        "the second one", "2", "b",
    })

    _CLASSIFIER_HISTORY_LIMIT = 8
    _CODE_RECOVERY_LOOKBACK = 10

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
        self._state_store = state_store

    async def handle(
        self,
        session_id: str,
        user_input: str,
    ) -> AsyncIterator[StreamEvent]:
        if not user_input or not user_input.strip():
            yield StreamEvent.error("Your message is empty. Please type something.")
            yield StreamEvent.done()
            return

        if len(user_input) > self._settings.max_input_chars:
            user_input = user_input[: self._settings.max_input_chars]
            logger.warning("Input truncated to %d chars.", self._settings.max_input_chars)

        smalltalk = _handle_smalltalk(user_input)
        if smalltalk is not None:
            self._memory.add_user(session_id, user_input)
            self._memory.add_assistant(session_id, smalltalk)
            logger.info("Small-talk reply sent.")
            yield StreamEvent.message(smalltalk)
            yield StreamEvent.done()
            return

        # ── Duplicate-solution check (state-agnostic) ────
        if _looks_like_solution(user_input):
            candidate_code = _extract_code_for_ingestion(user_input)
            if candidate_code and self._ingestion.is_duplicate(candidate_code):
                logger.info("Duplicate solution detected — short-circuiting.")
                self._state_store[session_id] = ConvState.IDLE
                self._memory.update_preferences(
                    session_id, pending_query=None, pending_code=None,
                )
                self._memory.add_user(session_id, user_input)
                msg = "I already have this solution — thanks!"
                self._memory.add_assistant(session_id, msg)
                yield StreamEvent.message(msg)
                yield StreamEvent.done()
                return

        resolved = self._try_resolve_clarification(session_id, user_input)
        if resolved is not None:
            classification, recovered_code, recovered_query = resolved
            self._memory.add_user(session_id, user_input)

            if classification.intent == Intent.EXPLAIN_CODE and not recovered_code:
                recovered_code = self._find_recent_code_in_history(session_id)

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

        # ── AWAITING_SOLUTION guard ──────────────
        if self._state_store.get(session_id) == ConvState.AWAITING_SOLUTION:
            if _looks_like_solution(user_input):
                code_to_ingest = _extract_code_for_ingestion(user_input)

                profile = self._memory.profile(session_id)
                pending_query = (profile.get("pending_query") or "").strip()
                pending_language = profile.get("language") or None

                logger.info(
                    "Ingesting user solution (chars=%d, lang=%s, query=%r)",
                    len(code_to_ingest), pending_language, pending_query[:80],
                )

                # Ingestion is the last hop in the learning loop — if it
                # fails, we still clear the FSM state so the user isn't
                # stuck waiting for a solution.
                try:
                    result = self._ingestion.ingest_solution(
                        session_id=session_id,
                        solution_text=code_to_ingest,
                        original_query=pending_query,
                        language=pending_language,
                    )
                    response_msg = result.message
                    logger.info("Ingestion result: %s", result.message)
                except Exception as exc:
                    logger.exception("Ingestion failed")
                    response_msg = (
                        "I couldn't store your solution right now — "
                        "something went wrong on my side. Your message is "
                        "saved in the conversation, but the knowledge base "
                        "wasn't updated."
                    )

                self._state_store[session_id] = ConvState.IDLE
                self._memory.update_preferences(
                    session_id, pending_query=None, pending_code=None,
                )
                self._memory.add_user(session_id, user_input)
                self._memory.add_assistant(session_id, response_msg)
                yield StreamEvent.message(response_msg)
                yield StreamEvent.done()
                return
            logger.info("AWAITING_SOLUTION cleared: not a solution.")
            self._state_store[session_id] = ConvState.IDLE

        self._memory.add_user(session_id, user_input)

        code = _extract_code(user_input)
        clean_query = _strip_fences(user_input) or user_input

        history = self._memory.full_history(
            session_id, limit=self._CLASSIFIER_HISTORY_LIMIT,
        )
        if history and history[-1].role == MessageRole.USER:
            history = history[:-1]

        try:
            classification = await self._classifier.classify(
                clean_query, code, history=history,
            )
        except Exception as exc:
            logger.exception("Classification failed")
            yield StreamEvent.error(f"Could not classify your request: {exc}")
            yield StreamEvent.done()
            return

        logger.info(
            "Classified as %s (conf=%.2f)",
            classification.intent, classification.confidence,
        )

        if classification.language:
            self._memory.set_language(session_id, classification.language)
        for fw in detect_frameworks(user_input):
            self._memory.add_framework(session_id, fw)

        if classification.intent == Intent.CLARIFY:
            self._state_store[session_id] = ConvState.AWAITING_CLARIFICATION
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

        if classification.intent == Intent.OUT_OF_SCOPE:
            msg = (
                "I'm specialized in programming tasks — code generation and "
                "explanation. Could you rephrase it as a coding question?"
            )
            self._memory.add_assistant(session_id, msg)
            yield StreamEvent.message(msg)
            yield StreamEvent.done()
            return

        if classification.intent == Intent.EXPLAIN_CODE and not code:
            recovered = self._find_recent_code_in_history(session_id)
            if recovered:
                code = recovered
                logger.info(
                    "Recovered %d chars of recent code from history.",
                    len(code),
                )

        async for event in self._dispatch(
            session_id=session_id,
            query=user_input,
            classification=classification,
            code=code,
        ):
            yield event

    async def _dispatch(
        self,
        *,
        session_id: str,
        query: str,
        classification: Classification,
        code: str | None,
    ) -> AsyncIterator[StreamEvent]:
        try:
            pipeline = self._router.route(classification.intent)
        except ValueError as exc:
            logger.error("Routing failed: %s", exc)
            yield StreamEvent.error("Internal routing error. Please try again.")
            yield StreamEvent.done()
            return

        yield StreamEvent.meta({
            "stage": "intent",
            "intent": classification.intent.value,
        })

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
                if event.kind == "state_change":
                    try:
                        new_state = ConvState(event.payload)
                        self._state_store[session_id] = new_state
                        if new_state == ConvState.AWAITING_SOLUTION:
                            self._memory.update_preferences(
                                session_id,
                                pending_query=query,
                                pending_code=code or "",
                            )
                            logger.info(
                                "AWAITING_SOLUTION set — pending_query=%r",
                                query[:80],
                            )
                    except ValueError:
                        logger.warning(
                            "Unknown state_change payload: %s", event.payload,
                        )
                yield event
        except Exception as exc:
            logger.exception("Pipeline execution failed")
            yield StreamEvent.error(f"Something went wrong: {exc}")
            yield StreamEvent.done()

    def _try_resolve_clarification(
        self,
        session_id: str,
        user_input: str,
    ) -> tuple[Classification, str | None, str] | None:
        if self._state_store.get(session_id) != ConvState.AWAITING_CLARIFICATION:
            return None

        ans = user_input.strip().lower()
        profile = self._memory.profile(session_id)
        pending_code = (profile.get("pending_code") or "").strip() or None
        pending_query = (profile.get("pending_query") or "").strip()

        self._state_store[session_id] = ConvState.IDLE
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

        logger.info(
            "Clarification answer %r not recognized; re-classifying.", ans,
        )
        return None

    def _find_recent_code_in_history(
        self,
        session_id: str,
        lookback: int = _CODE_RECOVERY_LOOKBACK,
    ) -> str | None:
        try:
            history = self._memory.full_history(session_id, limit=lookback)
        except Exception as exc:
            logger.warning("Could not read history for code recovery: %s", exc)
            return None

        for msg in reversed(history):
            if msg.role != MessageRole.ASSISTANT:
                continue
            block = _extract_code(msg.content)
            if block:
                return block
        return None