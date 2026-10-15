"""Embed → retrieve → filter → rerank.

Three-stage retrieval pipeline:
  1. Bi-encoder similarity search in Chroma (over-fetch candidates).
  2. Score threshold + MMR diversity filter.
  3. Cross-encoder rerank to pick the final top-k.
"""
from __future__ import annotations

import logging
import math

from app.core.exceptions import RetrievalError
from app.domain.models import RetrievedDoc
from app.domain.ports import EmbeddingProvider, VectorStore
from app.infrastructure.tools.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class Retriever:
    """Three-stage retrieval: fetch → filter → rerank."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        vectorstore: VectorStore,
        top_k: int = 4,
        min_score: float = 0.25,
        mmr_lambda: float = 0.7,
        reranker: CrossEncoderReranker | None = None,
        rerank_candidates: int = 20,
    ) -> None:
        self._embedder = embedder
        self._vdb = vectorstore
        self._top_k = top_k
        self._min_score = min_score
        self._mmr_lambda = mmr_lambda
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDoc]:
        k = top_k or self._top_k
        try:
            vector = self._embedder.embed_query(query)
            # Stage 1 — over-fetch candidates from the vector store.
            fetch_k = max(self._rerank_candidates, k)
            candidates = self._vdb.query(vector, top_k=fetch_k)
            logger.info("Fetched %d candidates (top score=%.3f).",
                        len(candidates), candidates[0].score if candidates else 0.0)

            # Stage 2 — score threshold.
            filtered = [d for d in candidates if d.score >= self._min_score]
            if not filtered:
                logger.info("All candidates below min_score=%.2f.", self._min_score)
                return []

            # Stage 2b — MMR diversity filter on the bi-encoder vectors.
            diversified = self._mmr(query, filtered, limit=fetch_k)

            # Stage 3 — cross-encoder rerank.
            if self._reranker is not None:
                final = self._reranker.rerank(query, diversified, top_k=k)
            else:
                final = diversified[:k]

            logger.info("Returning %d final docs.", len(final))
            return final
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

    # ── MMR ───────────────────────────────────────

    def _mmr(
        self,
        query: str,
        docs: list[RetrievedDoc],
        limit: int,
    ) -> list[RetrievedDoc]:
        """Maximal Marginal Relevance — balance relevance and diversity.

        Requires each doc to have an embedding. If embeddings are missing
        (e.g. Chroma didn't return them), skips MMR and returns input.
        """
        if not docs or all(d.embedding is None for d in docs):
            return docs
        if len(docs) <= limit:
            return docs

        selected: list[RetrievedDoc] = []
        remaining = list(docs)

        # Seed with the highest-scoring doc.
        remaining.sort(key=lambda d: d.score, reverse=True)
        selected.append(remaining.pop(0))

        while remaining and len(selected) < limit:
            best_doc = None
            best_score = -math.inf
            for d in remaining:
                if d.embedding is None:
                    continue
                relevance = d.score
                # Max similarity to already-selected docs.
                max_sim = 0.0
                for s in selected:
                    if s.embedding is None:
                        continue
                    sim = self._cosine(d.embedding, s.embedding)
                    max_sim = max(max_sim, sim)
                mmr = self._mmr_lambda * relevance - (1 - self._mmr_lambda) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_doc = d
            if best_doc is None:
                break
            selected.append(best_doc)
            remaining.remove(best_doc)

        return selected

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)