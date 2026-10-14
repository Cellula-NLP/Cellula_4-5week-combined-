"""Loads the HumanEval dataset and ingests canonical solutions into Chroma.

Dataset: https://huggingface.co/datasets/openai/openai_humaneval
Fields: task_id, prompt, canonical_solution, test, entry_point
"""
from __future__ import annotations

import hashlib
import logging

from app.core.exceptions import IngestionError
from app.domain.models import RetrievedDoc
from app.domain.ports import EmbeddingProvider, VectorStore
from app.infrastructure.loaders.chunker import CodeAwareChunker

logger = logging.getLogger(__name__)


class HumanEvalLoader:
    """One-shot loader: HumanEval test split → Chroma."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        vectorstore: VectorStore,
        chunker: CodeAwareChunker,
    ) -> None:
        self._embedder = embedder
        self._vdb = vectorstore
        self._chunker = chunker

    def load_and_ingest(self, split: str = "test") -> int:
        """Returns the number of chunks ingested."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise IngestionError(
                "datasets library is required. Run: pip install datasets"
            ) from exc

        logger.info("Downloading HumanEval split '%s' …", split)
        # NOTE: Hugging Face now requires the full namespace/name ID.
        dataset = load_dataset("openai/openai_humaneval", split=split)

        docs: list[RetrievedDoc] = []
        for row in dataset:
            task_id = row["task_id"]
            prompt = row.get("prompt", "")
            canonical = row.get("canonical_solution", "")
            entry_point = row.get("entry_point", "")

            # Ingest the full prompt + solution as one conceptual document,
            # then chunk it so retrieval can find function-level matches.
            body = (
                f"# Task: {task_id}\n"
                f"# Entry point: {entry_point}\n\n"
                f"{prompt}\n{canonical}"
            )
            for chunk in self._chunker.split(body, language="python"):
                content_hash = hashlib.sha256(chunk.text.encode()).hexdigest()
                docs.append(RetrievedDoc(
                    id=f"humaneval:{task_id}:{chunk.index}",
                    content=chunk.text,
                    metadata={
                        "source": "human_eval",
                        "task_id": task_id,
                        "entry_point": entry_point,
                        "language": "python",
                        "chunk_index": chunk.index,
                        "content_hash": content_hash,
                    },
                ))

        if not docs:
            logger.warning("No documents produced from HumanEval split '%s'.", split)
            return 0

        logger.info("Embedding %d chunks …", len(docs))
        vectors = self._embedder.embed_documents([d.content for d in docs])

        # Attach embeddings.
        enriched = [
            RetrievedDoc(
                id=d.id, content=d.content, metadata=d.metadata,
                score=d.score, embedding=v,
            )
            for d, v in zip(docs, vectors)
        ]

        logger.info("Upserting into Chroma …")
        self._vdb.upsert(enriched)
        logger.info("Ingested %d chunks from HumanEval.", len(enriched))
        return len(enriched)