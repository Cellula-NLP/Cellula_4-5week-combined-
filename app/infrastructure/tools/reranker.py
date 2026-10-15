"""Cross-encoder reranker — reorders retrieved documents by relevance.

A cross-encoder reads the (query, document) pair jointly, producing a
much more accurate relevance score than the bi-encoder cosine similarity
used during retrieval. We use it as a second-stage filter:

  bi-encoder → top-N candidates → cross-encoder → top-K final docs

Model: cross-encoder/ms-marco-MiniLM-L6-v2 (~22 MB, Apache-2.0, CPU-only)
"""
from __future__ import annotations

import logging

from app.domain.models import RetrievedDoc

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Wraps sentence-transformers' CrossEncoder with lazy loading.

    The CrossEncoder is loaded on first use so app startup stays fast.
    If loading fails, ``rerank`` returns the input unchanged (graceful).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        enabled: bool = True,
    ) -> None:
        self._model_name = model_name
        self._enabled = enabled
        self._model = None
        self._load_failed = False

    # ── Public API ────────────────────────────────

    def rerank(
        self,
        query: str,
        docs: list[RetrievedDoc],
        top_k: int = 4,
    ) -> list[RetrievedDoc]:
        """Return the top-k docs reordered by cross-encoder relevance.

        On any failure, returns ``docs[:top_k]`` unchanged so the pipeline
        never breaks because of a reranker issue.
        """
        if not self._enabled or not docs:
            return docs[:top_k]

        model = self._ensure_loaded()
        if model is None:
            return docs[:top_k]

        try:
            pairs = [[query, d.content] for d in docs]
            scores = model.predict(pairs, show_progress_bar=False)
            scored = sorted(
                zip(scores, docs),
                key=lambda t: float(t[0]),
                reverse=True,
            )
            logger.info(
                "Reranked %d candidates → top-%d (best score=%.3f).",
                len(docs), top_k, float(scored[0][0]) if scored else 0.0,
            )
            return [d for _, d in scored[:top_k]]
        except Exception as exc:
            logger.warning("Reranker inference failed (%s); returning original order.", exc)
            return docs[:top_k]

    # ── Internals ─────────────────────────────────

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            logger.warning("sentence-transformers CrossEncoder unavailable: %s", exc)
            self._load_failed = True
            return None

        try:
            logger.info("Loading reranker %s …", self._model_name)
            self._model = CrossEncoder(self._model_name, max_length=512)
            logger.info("Reranker loaded.")
            return self._model
        except Exception as exc:
            logger.warning("Failed to load reranker %s: %s", self._model_name, exc)
            self._load_failed = True
            return None