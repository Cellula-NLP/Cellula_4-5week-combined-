"""ChromaDB-backed vector store with graceful degradation.

If Chroma is unavailable at startup, falls back to an in-memory store
so the rest of the app remains functional (degraded mode).
"""
from __future__ import annotations

import logging
from typing import Sequence

from app.core.exceptions import RetrievalError
from app.domain.models import RetrievedDoc
from app.domain.ports import VectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """Persistent Chroma collection wrapper."""

    def __init__(self, persist_path: str, collection_name: str) -> None:
        self._persist_path = persist_path
        self._collection_name = collection_name
        self._client = None
        self._collection = None
        self._in_memory: dict[str, RetrievedDoc] = {}  # fallback
        self._degraded = False
        self._init_client()

    # ── Init ──────────────────────────────────────

    def _init_client(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=self._persist_path,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Chroma ready. Collection '%s' has %d docs.",
                self._collection_name,
                self._collection.count(),
            )
        except Exception as exc:
            logger.error("Chroma init failed (%s). Falling back to in-memory store.", exc)
            self._degraded = True

    # ── VectorStore API ───────────────────────────

    def upsert(self, docs: Sequence[RetrievedDoc]) -> list[str]:
        if not docs:
            return []
        if self._degraded:
            for d in docs:
                self._in_memory[d.id] = d
            return [d.id for d in docs]

        try:
            self._collection.upsert(
                ids=[d.id for d in docs],
                documents=[d.content for d in docs],
                metadatas=[d.metadata for d in docs],
                embeddings=[d.embedding for d in docs] if docs[0].embedding else None,
            )
            return [d.id for d in docs]
        except Exception as exc:
            raise RetrievalError(f"Chroma upsert failed: {exc}") from exc

    def query(
        self,
        embedding: list[float],
        top_k: int = 4,
        filters: dict | None = None,
    ) -> list[RetrievedDoc]:
        if self._degraded:
            return self._in_memory_query(embedding, top_k)

        try:
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=filters or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise RetrievalError(f"Chroma query failed: {exc}") from exc

        docs: list[RetrievedDoc] = []
        ids = result.get("ids", [[]])[0]
        contents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for i, doc_id in enumerate(ids):
            # Chroma returns cosine distance; convert to similarity score.
            dist = distances[i] if i < len(distances) else 1.0
            score = max(0.0, 1.0 - dist)
            docs.append(RetrievedDoc(
                id=doc_id,
                content=contents[i] if i < len(contents) else "",
                metadata=metadatas[i] if i < len(metadatas) else {},
                score=score,
            ))
        return docs

    def count(self) -> int:
        if self._degraded:
            return len(self._in_memory)
        try:
            return self._collection.count()
        except Exception:
            return 0

    def query_by_hash(self, content_hash: str) -> list[RetrievedDoc]:
        if self._degraded:
            return [d for d in self._in_memory.values()
                    if d.metadata.get("content_hash") == content_hash]
        try:
            result = self._collection.get(
                where={"content_hash": content_hash},
                include=["documents", "metadatas"],
            )
        except Exception:
            return []
        out: list[RetrievedDoc] = []
        for i, doc_id in enumerate(result.get("ids", [])):
            out.append(RetrievedDoc(
                id=doc_id,
                content=result["documents"][i],
                metadata=result["metadatas"][i],
            ))
        return out

    # ── Fallback helpers ──────────────────────────

    def _in_memory_query(self, embedding: list[float], top_k: int) -> list[RetrievedDoc]:
        """Brute-force cosine similarity for the degraded path."""
        import math
        scored: list[tuple[float, RetrievedDoc]] = []
        for doc in self._in_memory.values():
            if not doc.embedding:
                continue
            dot = sum(a * b for a, b in zip(embedding, doc.embedding))
            norm_a = math.sqrt(sum(a * a for a in embedding)) or 1.0
            norm_b = math.sqrt(sum(b * b for b in doc.embedding)) or 1.0
            sim = dot / (norm_a * norm_b)
            scored.append((sim, doc))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            RetrievedDoc(id=d.id, content=d.content, metadata=d.metadata, score=s)
            for s, d in scored[:top_k]
        ]