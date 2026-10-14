"""SQLite-backed conversation memory with profile persistence."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.exceptions import MemoryError
from app.domain.enums import MessageRole
from app.domain.models import Message
from app.domain.ports import MemoryStore

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS profiles (
    session_id TEXT PRIMARY KEY,
    data TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
);
"""


class SQLiteMemoryStore(MemoryStore):
    """Thread-safe SQLite store for chat history and user profiles."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        try:
            with self._lock, self._conn() as conn:
                conn.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            raise MemoryError(f"Failed to init memory DB: {exc}") from exc

    # ── MemoryStore API ───────────────────────────

    def append(self, session_id: str, msg: Message) -> None:
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    "INSERT INTO messages (session_id, role, content, created_at, meta) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        session_id,
                        msg.role.value,
                        msg.content,
                        msg.created_at.isoformat(),
                        json.dumps(msg.meta),
                    ),
                )
        except sqlite3.Error as exc:
            raise MemoryError(f"Failed to append message: {exc}") from exc

    def history(self, session_id: str, limit: int | None = None) -> list[Message]:
        sql = (
            "SELECT role, content, created_at, meta FROM messages "
            "WHERE session_id = ? ORDER BY id DESC"
        )
        params: list[Any] = [session_id]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)

        with self._lock, self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        # Reverse so oldest-first.
        out: list[Message] = []
        for row in reversed(rows):
            try:
                role = MessageRole(row["role"])
            except ValueError:
                role = MessageRole.USER
            out.append(Message(
                role=role,
                content=row["content"],
                created_at=datetime.fromisoformat(row["created_at"]),
                meta=json.loads(row["meta"] or "{}"),
            ))
        return out

    def profile(self, session_id: str) -> dict:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT data FROM profiles WHERE session_id = ?", (session_id,)
            ).fetchone()
        return json.loads(row["data"]) if row else {}

    def update_profile(self, session_id: str, patch: dict) -> None:
        current = self.profile(session_id)
        current.update(patch)
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    "INSERT INTO profiles (session_id, data, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET data=excluded.data, "
                    "updated_at=excluded.updated_at",
                    (session_id, json.dumps(current), now),
                )
        except sqlite3.Error as exc:
            raise MemoryError(f"Failed to update profile: {exc}") from exc

    def clear(self, session_id: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM profiles WHERE session_id = ?", (session_id,))