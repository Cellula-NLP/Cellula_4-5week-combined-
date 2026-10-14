"""Code-aware chunking that keeps function/class boundaries intact."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    index: int


class CodeAwareChunker:
    """Splits code on top-level `def` / `class` boundaries, with size caps."""

    def __init__(self, max_chars: int = 1500, overlap_chars: int = 100) -> None:
        self._max_chars = max_chars
        self._overlap = overlap_chars

    def split(self, text: str, language: str = "python") -> list[Chunk]:
        if not text.strip():
            return []

        if language == "python":
            blocks = self._split_python(text)
        else:
            blocks = [text]

        # Merge small blocks, split oversized ones.
        merged = self._merge_blocks(blocks)
        final: list[Chunk] = []
        for block in merged:
            for sub in self._hard_split(block):
                final.append(Chunk(text=sub, index=len(final)))
        return final

    # ── Internals ─────────────────────────────────

    @staticmethod
    def _split_python(text: str) -> list[str]:
        """Split on top-level `def ` / `class ` declarations."""
        pattern = re.compile(r"^(?=(?:def |class |async def ))", re.MULTILINE)
        parts = pattern.split(text)
        return [p for p in parts if p.strip()]

    def _merge_blocks(self, blocks: list[str]) -> list[str]:
        """Merge adjacent small blocks up to max_chars."""
        out: list[str] = []
        buf = ""
        for block in blocks:
            if len(buf) + len(block) <= self._max_chars:
                buf += ("\n" if buf else "") + block
            else:
                if buf:
                    out.append(buf)
                buf = block
        if buf:
            out.append(buf)
        return out

    def _hard_split(self, block: str) -> list[str]:
        """Fallback: split large blocks by character count with overlap."""
        if len(block) <= self._max_chars:
            return [block]
        chunks: list[str] = []
        start = 0
        while start < len(block):
            end = start + self._max_chars
            chunks.append(block[start:end])
            start = end - self._overlap if end < len(block) else len(block)
        return chunks