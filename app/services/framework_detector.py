"""Detect programming frameworks mentioned in user messages.
"""
from __future__ import annotations

import re

# Order matters only for iteration; matching is done by regex word boundary.
_FRAMEWORKS: dict[str, re.Pattern] = {
    # Python
    "Flask":       re.compile(r"\bflask\b", re.I),
    "Django":      re.compile(r"\bdjango\b", re.I),
    "FastAPI":     re.compile(r"\bfastapi\b", re.I),
    "Streamlit":   re.compile(r"\bstreamlit\b", re.I),
    "PyTorch":     re.compile(r"\b(pytorch|torch)\b", re.I),
    "TensorFlow":  re.compile(r"\btensorflow\b", re.I),
    "Keras":       re.compile(r"\bkeras\b", re.I),
    "Scikit-learn":re.compile(r"\b(scikit[- ]?learn|sklearn)\b", re.I),
    "Pandas":      re.compile(r"\bpandas\b", re.I),
    "NumPy":       re.compile(r"\bnumpy\b", re.I),
    "SQLAlchemy":  re.compile(r"\bsqlalchemy\b", re.I),
    "Pydantic":    re.compile(r"\bpydantic\b", re.I),
    "Celery":      re.compile(r"\bcelery\b", re.I),
    # JavaScript / TypeScript
    "React":       re.compile(r"\breact(\.js)?\b", re.I),
    "Next.js":     re.compile(r"\bnext\.?js\b", re.I),
    "Vue":         re.compile(r"\bvue(\.js)?\b", re.I),
    "Angular":     re.compile(r"\bangular\b", re.I),
    "Svelte":      re.compile(r"\bsvelte(kit)?\b", re.I),
    "Express":     re.compile(r"\bexpress(\.js)?\b", re.I),
    "Node.js":     re.compile(r"\bnode(\.js)?\b", re.I),
    "NestJS":      re.compile(r"\bnest(\.?js)?\b", re.I),
    # Java / Kotlin
    "Spring":      re.compile(r"\bspring(\s*boot)?\b", re.I),
    "Hibernate":   re.compile(r"\bhibernate\b", re.I),
    "Ktor":        re.compile(r"\bktor\b", re.I),
    # Ruby / PHP / Go / Rust
    "Rails":       re.compile(r"\brails\b", re.I),
    "Laravel":     re.compile(r"\blaravel\b", re.I),
    "Gin":         re.compile(r"\bgin\b", re.I),
    "Actix":       re.compile(r"\bactix\b", re.I),
}


def detect_frameworks(text: str) -> list[str]:
    """Return a list of framework names mentioned in ``text``.

    Only returns frameworks that appear with word boundaries, so prose
    like "reacting to input" won't be mistaken for "React".
    """
    if not text:
        return []
    return [name for name, pat in _FRAMEWORKS.items() if pat.search(text)]