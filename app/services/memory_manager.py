"""High-level memory facade used by the orchestrator."""
from __future__ import annotations

import logging

from app.domain.models import Message
from app.domain.ports import MemoryStore

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self, store: MemoryStore, window: int = 12) -> None:
        self._store = store
        self._window = window

    def add_user(self, session_id: str, content: str) -> None:
        self._store.append(session_id, Message(role="user", content=content))

    def add_assistant(self, session_id: str, content: str, meta: dict | None = None) -> None:
        self._store.append(session_id, Message(role="assistant", content=content, meta=meta or {}))

    def context(self, session_id: str) -> list[Message]:
        return self._store.history(session_id, limit=self._window)

    def profile(self, session_id: str) -> dict:
        return self._store.profile(session_id)

    def update_preferences(self, session_id: str, **kwargs) -> None:
        self._store.update_profile(session_id, kwargs)

    def clear(self, session_id: str) -> None:
        self._store.clear(session_id)