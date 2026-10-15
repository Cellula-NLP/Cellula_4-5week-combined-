"""High-level memory facade used by the orchestrator."""
from __future__ import annotations

import logging

from app.domain.models import Message
from app.domain.ports import MemoryStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """Thin, typed wrapper around the MemoryStore port.

    Provides convenience methods for the preferences required by the
    task (language, framework) so callers don't juggle dicts directly.
    """

    def __init__(self, store: MemoryStore, window: int = 12) -> None:
        self._store = store
        self._window = window

    # ── Turns ─────────────────────────────────────

    def add_user(self, session_id: str, content: str, meta: dict | None = None) -> None:
        self._store.append(
            session_id,
            Message(role="user", content=content, meta=meta or {}),
        )

    def add_assistant(self, session_id: str, content: str, meta: dict | None = None) -> None:
        self._store.append(
            session_id,
            Message(role="assistant", content=content, meta=meta or {}),
        )

    def context(self, session_id: str) -> list[Message]:
        return self._store.history(session_id, limit=self._window)

    def full_history(self, session_id: str, limit: int | None = None) -> list[Message]:
        return self._store.history(session_id, limit=limit)

    # ── Profile / preferences ─────────────────────

    def profile(self, session_id: str) -> dict:
        return self._store.profile(session_id)

    def update_preferences(self, session_id: str, **kwargs) -> None:
        self._store.update_profile(session_id, kwargs)

    def set_language(self, session_id: str, language: str | None) -> None:
        """Set the user's preferred programming language (or clear it)."""
        self._store.update_profile(session_id, {"language": language or None})

    def add_framework(self, session_id: str, framework: str) -> None:
        """Append a framework to the preference list (deduped)."""
        current = self._store.profile(session_id)
        frameworks = list(current.get("frameworks") or [])
        if framework not in frameworks:
            frameworks.append(framework)
            self._store.update_profile(session_id, {"frameworks": frameworks})
            logger.info("Added framework preference %r for session %s", framework, session_id[:8])

    def remove_framework(self, session_id: str, framework: str) -> None:
        current = self._store.profile(session_id)
        frameworks = [f for f in (current.get("frameworks") or []) if f != framework]
        self._store.update_profile(session_id, {"frameworks": frameworks})

    def clear(self, session_id: str) -> None:
        self._store.clear(session_id)