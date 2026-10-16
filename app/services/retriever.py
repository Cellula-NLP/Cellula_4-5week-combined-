"""Embed → retrieve → filter → rerank.

Three-stage retrieval pipeline:
  1. Bi-encoder similarity search in Chroma (over-fetch candidates).
     Also, fetch user-provided solutions that match the *current* query
     (their ``original_query`` must be close) so a relevant taught
     solution can't be missed due to short-snippet embedding noise.
  2. Score threshold filter — never drops user_feedback docs that
     survived the similarity gate.
  3. Cross-encoder rerank to pick the final top-k. Matching
     user_feedback docs are always prepended to the output.
"""
from __future__ import annotations

import logging

from app.core.exceptions import RetrievalError
from app.domain.models import RetrievedDoc
from app.domain.ports import EmbeddingProvider, VectorStore
from app.infrastructure.tools.reranker import CrossEncoderReranker
from app.services.query_similarity import queries_match

logger = logging.getLogger(__name__)


class Retriever:
    """Three-stage retrieval: fetch → filter → rerank."""

    # Max number of user-provided solutions to force-include.
    _MAX_FEEDBACK_DOCS = 3

    def __init__(
        self,
        embedder: EmbeddingProvider,
        vectorstore: VectorStore,
        top_k: int = 4,
        min_score: float = 0.25,
        reranker: CrossEncoderReranker | None = None,
        rerank_candidates: int = 20,
    ) -> None:
        self._embedder = embedder
        self._vdb = vectorstore
        self._top_k = top_k
        self._min_score = min_score
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDoc]:
        k = top_k or self._top_k
        try:
            vector = self._embedder.embed_query(query)

            # ── Stage 1a — over-fetch general candidates ──
            fetch_k = max(self._rerank_candidates, k)
            candidates = list(self._vdb.query(vector, top_k=fetch_k))
            logger.info(
                "Fetched %d candidates (top score=%.3f).",
                len(candidates), candidates[0].score if candidates else 0.0,
            )

            # ── Stage 1b — matching user-provided solutions ──
            # Only include feedback docs whose original_query matches the
            # current request. Unrelated taught solutions must NOT be
            # forced into the context — they would crowd out genuinely
            # relevant results.
            feedback_docs = self._fetch_matching_feedback_docs(vector, query)
            if feedback_docs:
                logger.info(
                    "Fetched %d matching user-feedback docs.", len(feedback_docs),
                )
                existing_ids = {d.id for d in candidates}
                for d in feedback_docs:
                    if d.id not in existing_ids:
                        candidates.append(d)
                        existing_ids.add(d.id)

            if not candidates:
                return []

            # ── Stage 2 — score threshold (never drops feedback docs) ──
            filtered = [
                d for d in candidates
                if d.score >= self._min_score
                or d.metadata.get("source") == "user_feedback"
            ]
            if not filtered:
                logger.info(
                    "All candidates below min_score=%.2f.", self._min_score,
                )
                return []

            # ── Stage 3 — rerank non-feedback; always prepend feedback ──
            regular = [
                d for d in filtered
                if d.metadata.get("source") != "user_feedback"
            ]
            feedback = [
                d for d in filtered
                if d.metadata.get("source") == "user_feedback"
            ][: self._MAX_FEEDBACK_DOCS]

            if self._reranker is not None and regular:
                reranked = self._reranker.rerank(query, regular, top_k=k)
            else:
                reranked = regular[:k]

            final = (feedback + reranked)[:k]
            logger.info(
                "Returning %d final docs (%d feedback, %d reranked).",
                len(final), len(feedback), len(final) - len(feedback),
            )
            return final
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

    # ── Internals ─────────────────────────────────

    def _fetch_matching_feedback_docs(
        self,
        vector: list[float],
        query: str,
    ) -> list[RetrievedDoc]:
        """Fetch user-feedback docs whose original_query matches ``query``.

        Falls back to an empty list if the vector store doesn't support
        metadata filters (e.g. the in-memory degraded mode).
        """
        try:
            docs = self._vdb.query(
                vector,
                top_k=self._MAX_FEEDBACK_DOCS,
                filters={"source": "user_feedback"},
            )
        except Exception as exc:
            logger.debug("Feedback-filtered query skipped: %s", exc)
            return []

        kept: list[RetrievedDoc] = []
        for d in docs:
            original = (d.metadata.get("original_query") or "").strip()
            if not original:
                # No signal about what this doc is for → drop it.
                continue
            if queries_match(query, original):
                kept.append(d)
            else:
                logger.debug(
                    "Dropping unrelated feedback doc %s "
                    "(query=%r vs original_query=%r).",
                    d.id, query[:60], original[:60],
                )
        return kept