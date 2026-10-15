from __future__ import annotations

import logging

from app.domain.enums import MessageRole
from app.domain.models import Message

logger = logging.getLogger(__name__)

# A query of at least this many words is treated as standalone.
_MIN_STANDALONE_WORDS = 12

# Below this many words, we add a domain hint to improve retrieval.
_SHORT_QUERY_WORDS = 6

# Max chars of prior context to include.
_MAX_CONTEXT_CHARS = 400

# Words at the start of a query that indicate a follow-up, not a new topic.
_FOLLOWUP_MARKERS = (
    "now ", "add ", "also ", "then ", "instead ", "but ",
    "make it ", "change ", "remove ", "modify ", "improve ",
    "refactor ", "convert ", "rewrite ", "fix ", "update ",
    "use ", "what about ", "can you ", "could you ",
    "why ", "how about ", "and ",
)


class QueryRewriter:
    """Expand short follow-up queries with context from memory.

    Two enrichment strategies:
      1. Follow-ups ("now add error handling") → prepend prior user message.
      2. Short standalone queries → prepend a language-aware hint
         (uses the classifier's detected language, not a hardcoded one).
    """

    def __init__(self, memory, history_window: int = 6) -> None:
        self._memory = memory
        self._window = history_window

    def rewrite(
        self,
        session_id: str,
        query: str,
        language: str | None = None,
    ) -> str:
        """Return an enriched query suitable for vector retrieval.

        ``language`` comes from the classifier (e.g. "python", "javascript").
        It's used to hint the retrieval, not to force a language.
        """
        query = (query or "").strip()
        if not query:
            return query

        # ── Follow-up path: prepend the previous user turn ──
        if not self._is_standalone(query):
            prior = self._prior_user_message(session_id, query)
            if prior:
                enriched = f"{prior} — {query}"
                logger.info("Rewrote follow-up: %r", enriched[:120])
                return enriched
            # No prior context — fall through to short-query handling.

        # ── Short-query path: add a language hint if we know one ──
        if len(query.split()) < _SHORT_QUERY_WORDS:
            hinted = self._add_domain_hint(query, language)
            if hinted != query:
                logger.info("Enriched short query: %r", hinted)
            return hinted

        return query

    # ── Internals ──────────────────────────────────

    @staticmethod
    def _add_domain_hint(query: str, language: str | None) -> str:
        """Prefix the query with a light domain hint.

        Uses the detected language if available; otherwise a generic
        'code' hint. Never hardcodes Python.
        """
        if language:
            return f"{language} code: {query}"
        return f"code: {query}"

    @staticmethod
    def _is_standalone(query: str) -> bool:
        words = query.split()
        if len(words) >= _MIN_STANDALONE_WORDS:
            return True
        ql = query.lower()
        return not any(ql.startswith(m) for m in _FOLLOWUP_MARKERS)

    def _prior_user_message(self, session_id: str, current: str) -> str | None:
        """Get the last user message that isn't the current one."""
        history = self._memory.full_history(session_id, limit=self._window)
        user_msgs = [m for m in history if m.role == MessageRole.USER]
        if not user_msgs:
            return None
        for m in reversed(user_msgs):
            if m.content.strip() != current.strip():
                return m.content[:_MAX_CONTEXT_CHARS]
        return None