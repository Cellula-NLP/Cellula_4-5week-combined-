"""Shared query similarity helper.

Used by:
  • the retriever, to decide whether a document's ``original_query``
    is close enough to the current request to be force-included.
  • the relevance evaluator, to short-circuit an LLM call when a
    matching document is already present.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

# Ratio at which two normalized queries are considered "the same".
DEFAULT_MATCH_THRESHOLD = 0.75


def normalize(s: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation."""
    return re.sub(r"\s+", " ", (s or "").lower()).strip().rstrip(".!?,")


def queries_match(
    a: str,
    b: str,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> bool:
    """True if two queries are identical or very similar.

    Empty inputs never match (they'd cause false positives).
    """
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return False
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold