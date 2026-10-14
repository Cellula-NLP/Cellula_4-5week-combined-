"""Ingest user-provided solutions into the vector store (human feedback learning)."""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

from app.core.exceptions import IngestionError
from app.domain.models import IngestResult, RetrievedDoc
from app.domain.ports import EmbeddingProvider, VectorStore
from app.infrastructure.loaders.chunker import CodeAwareChunker

logger = logging.getLogger(__name__)


class IngestionService:
    """Chunk → embed → upsert user solutions with deduplication."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        vectorstore: VectorStore,
        chunker: CodeAwareChunker,
    ) -> None:
        self._embedder = embedder
        self._vdb = vectorstore
        self._chunker = chunker

    def ingest_solution(
        self,
        session_id: str,
        solution_text: str,
        original_query: str,
        language: str | None = None,
    ) -> IngestResult:
        text = solution_text.strip()
        if len(text) < 20:
            return IngestResult(False, "Solution is too short to be useful.")

        content_hash = hashlib.sha256(text.encode()).hexdigest()

        # Deduplication check.
        if self._vdb.query_by_hash(content_hash):
            return IngestResult(False, "I already have this solution — thanks!")

        try:
            chunks = self._chunker.split(text, language=language or "python")
            if not chunks:
                return IngestResult(False, "Could not extract any chunk from your solution.")

            vectors = self._embedder.embed_documents([c.text for c in chunks])

            docs = [
                RetrievedDoc(
                    id=f"userfb:{content_hash}:{c.index}",
                    content=c.text,
                    metadata={
                        "source": "user_feedback",
                        "session_id": session_id,
                        "original_query": original_query,
                        "language": language or "unknown",
                        "content_hash": content_hash,
                        "chunk_index": c.index,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    embedding=v,
                )
                for c, v in zip(chunks, vectors)
            ]
            self._vdb.upsert(docs)
            logger.info("Ingested %d chunks from user feedback.", len(docs))
            return IngestResult(
                True,
                f"Learned {len(docs)} chunk(s) from your solution. Thanks!",
                chunks_ingested=len(docs),
            )
        except Exception as exc:
            logger.error("Ingestion failed: %s", exc)
            raise IngestionError(f"Ingestion failed: {exc}") from exc