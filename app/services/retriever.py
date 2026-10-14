"""Embed + query the vector store for relevant documents."""
from __future__ import annotations

import logging

from app.core.exceptions import RetrievalError
from app.domain.models import RetrievedDoc
from app.domain.ports import EmbeddingProvider, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Thin orchestration around embedder + vector store."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        vectorstore: VectorStore,
        top_k: int = 4,
    ) -> None:
        self._embedder = embedder
        self._vdb = vectorstore
        self._top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDoc]:
        k = top_k or self._top_k
        try:
            vector = self._embedder.embed_query(query)
            docs = self._vdb.query(vector, top_k=k)
            logger.info("Retrieved %d docs (top score=%.3f).",
                        len(docs), docs[0].score if docs else 0.0)
            return docs
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            raise RetrievalError(f"Retrieval failed: {exc}") from exc