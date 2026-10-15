"""AST-based code chunking — structurally aware, not regex-based.

Uses Python's built-in ``ast`` module to split code along real syntax
boundaries (module docstring, imports, top-level functions, classes).
Falls back to a line-window chunker for non-Python or syntactically
invalid code.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    index: int


class CodeAwareChunker:
    """Splits code using the AST when possible, regex-free.

    Strategy for Python:
      1. Parse the source with ``ast``.
      2. Group top-level nodes into "units":
         • Module docstring + imports → one unit
         • Each top-level FunctionDef / AsyncFunctionDef / ClassDef → one unit
      3. Merge small adjacent units up to ``max_chars``.
      4. Hard-split any oversized unit with overlap.

    Fallback (non-Python or SyntaxError): character-window split.
    """

    def __init__(self, max_chars: int = 1500, overlap_chars: int = 100) -> None:
        self._max_chars = max_chars
        self._overlap = overlap_chars

    # ── Public API ────────────────────────────────

    def split(self, text: str, language: str = "python") -> list[Chunk]:
        if not text.strip():
            return []

        if language == "python":
            blocks = self._split_python_ast(text)
        else:
            blocks = [text]

        merged = self._merge_blocks(blocks)
        final: list[Chunk] = []
        for block in merged:
            for sub in self._hard_split(block):
                final.append(Chunk(text=sub, index=len(final)))
        return final

    # ── Python AST path ───────────────────────────

    def _split_python_ast(self, source: str) -> list[str]:
        """Extract top-level units using ast, preserving exact source text."""
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            logger.debug("AST parse failed (%s); falling back to text split.", exc)
            return [source]

        lines = source.splitlines(keepends=True)
        units: list[str] = []
        pending_start: int | None = None  # start of a header/import block

        # ast nodes have 1-based line numbers; convert to 0-based slices.
        def _node_text(node: ast.AST) -> str:
            start = (getattr(node, "lineno", 1) - 1)
            # Include decorators.
            for dec in getattr(node, "decorator_list", []) or []:
                start = min(start, getattr(dec, "lineno", start + 1) - 1)
            end = getattr(node, "end_lineno", None)
            if end is None:
                return ""
            return "".join(lines[start:end])

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Flush any pending header block first.
                if pending_start is not None:
                    block = "".join(lines[pending_start:node.lineno - 1]).rstrip()
                    if block.strip():
                        units.append(block)
                    pending_start = None
                body = _node_text(node)
                if body.strip():
                    units.append(body)
            else:
                # Group docstring / imports / assignments into one header block.
                if pending_start is None:
                    pending_start = getattr(node, "lineno", 1) - 1

        # Flush trailing header block.
        if pending_start is not None:
            tail = "".join(lines[pending_start:]).rstrip()
            if tail.strip():
                units.append(tail)

        return units or [source]

    # ── Shared merge / split ──────────────────────

    def _merge_blocks(self, blocks: list[str]) -> list[str]:
        """Merge adjacent small blocks up to max_chars."""
        out: list[str] = []
        buf = ""
        for block in blocks:
            if len(buf) + len(block) <= self._max_chars:
                buf += ("\n\n" if buf else "") + block
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