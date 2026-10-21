"""Shared query similarity helper for RAG."""
from __future__ import annotations
import re
from difflib import SequenceMatcher

DEFAULT_MATCH_THRESHOLD = 0.75

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip().rstrip(".!?,")

def queries_match(a: str, b: str, threshold: float = DEFAULT_MATCH_THRESHOLD) -> bool:
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return False
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold