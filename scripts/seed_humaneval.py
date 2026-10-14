"""One-time ingestion of the HumanEval dataset into Chroma.

Usage:
    python scripts/seed_humaneval.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.infrastructure.embeddings.sentence_transformer import SentenceTransformerEmbedder
from app.infrastructure.vectorstore.chroma_store import ChromaVectorStore
from app.infrastructure.loaders.chunker import CodeAwareChunker
from app.infrastructure.loaders.humaneval_loader import HumanEvalLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> int:
    settings = get_settings()
    print(f"Embedding model: {settings.embedder_model}")
    print(f"Chroma path:     {settings.chroma_path}")
    print(f"Collection:      {settings.chroma_collection}")

    embedder = SentenceTransformerEmbedder(settings.embedder_model)
    vdb = ChromaVectorStore(settings.chroma_path, settings.chroma_collection)
    chunker = CodeAwareChunker()

    loader = HumanEvalLoader(embedder, vdb, chunker)
    count = loader.load_and_ingest(split="test")

    print(f"\n✅ Ingested {count} chunks. Total in collection: {vdb.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())