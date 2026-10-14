"""Local, free embedding provider using BAAI/bge-small-en-v1.5.

Runs entirely on CPU — no API key, no network after first download.
"""
from __future__ import annotations

import logging
from typing import Sequence

from app.core.exceptions import ConfigError
from app.domain.ports import EmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Wraps sentence-transformers with lazy loading and query prefix."""

    # BGE models expect this prefix on queries for best retrieval quality.
    _QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model = None  # lazy

    def _ensure_loaded(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ConfigError(
                    "sentence-transformers is not installed. Run: pip install -e ."
                ) from exc
            logger.info("Loading embedding model %s …", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded.")
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_loaded()
        vectors = model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_loaded()
        vec = model.encode(
            [self._QUERY_PREFIX + text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vec.tolist()

    @property
    def dimension(self) -> int:
        return 384  # bge-small-en-v1.5 is fixed at 384