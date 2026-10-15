"""Code-specialized embedding provider with automatic fallback.

Primary: jinaai/jina-embeddings-v2-base-code (768 dims, 30 programming languages)
Fallback: BAAI/bge-small-en-v1.5 (384 dims, general English)

The fallback ensures the app still works if the primary model can't be
downloaded or fails to load (e.g. on constrained environments).
"""
from __future__ import annotations

import logging
from typing import Sequence

from app.core.exceptions import ConfigError
from app.domain.ports import EmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Wraps sentence-transformers with lazy loading and fallback.

    Loads a code-specialized model first; if that fails, falls back to
    a general-purpose English model. The ``dimension`` property reports
    whichever model actually loaded.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-base-code",
        fallback_model: str = "BAAI/bge-small-en-v1.5",
        query_prefix: str = "",
        expected_dim: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._fallback_model = fallback_model
        self._query_prefix = query_prefix
        self._expected_dim = expected_dim
        self._model = None
        self._loaded_model_name: str | None = None
        self._dim: int | None = None

    # ── Loading ───────────────────────────────────

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model

        # Try primary (code-specialized) model first.
        try:
            self._model = self._load_model(self._model_name)
            self._loaded_model_name = self._model_name
            logger.info("Loaded primary embedding model: %s", self._model_name)
        except Exception as exc:
            logger.warning(
                "Primary embedder %s failed (%s). Falling back to %s.",
                self._model_name, exc, self._fallback_model,
            )
            try:
                self._model = self._load_model(self._fallback_model)
                self._loaded_model_name = self._fallback_model
                logger.info("Loaded fallback embedding model: %s", self._fallback_model)
            except Exception as exc2:
                raise ConfigError(
                    f"Both embedders failed. Primary: {exc}. Fallback: {exc2}"
                ) from exc2

        self._dim = int(self._model.get_sentence_embedding_dimension())
        logger.info(
            "Embedder ready: %s (%d dims)",
            self._loaded_model_name, self._dim,
        )
        return self._model

    @staticmethod
    def _load_model(name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigError(
                "sentence-transformers is not installed. Run: pip install -e ."
            ) from exc

        logger.info("Loading embedding model %s …", name)
        # Jina models use custom architecture — trust_remote_code is required.
        # BGE models load normally; passing the kwarg is harmless for them.
        try:
            return SentenceTransformer(name, trust_remote_code=True)
        except TypeError:
            # Older ST versions don't accept trust_remote_code at init.
            return SentenceTransformer(name)

    # ── EmbeddingProvider API ─────────────────────

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_loaded()
        vectors = model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=16,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_loaded()
        payload = f"{self._query_prefix}{text}" if self._query_prefix else text
        vec = model.encode(
            [payload],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vec.tolist()

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    @property
    def model_name(self) -> str:
        """Return the model that actually loaded (primary or fallback)."""
        if self._loaded_model_name is None:
            self._ensure_loaded()
        return self._loaded_model_name or self._model_name