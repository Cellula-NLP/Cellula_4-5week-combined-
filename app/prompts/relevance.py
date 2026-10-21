"""Prompt for the relevance evaluator."""
from __future__ import annotations
from app.domain.models import RetrievedDoc

_SYSTEM = """You evaluate whether retrieved coding documents are relevant.
Respond with STRICT JSON only (no markdown, no prose):
{
  "verdict": "relevant" | "not_relevant",
  "score": 0.0-1.0,
  "reason": "one short sentence"
}
"""

def build_relevance_prompt(query: str, docs: list[RetrievedDoc]) -> list[dict]:
    snippets = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        snippet = d.content[:900]
        snippets.append(f"[{i}] source={src}\n{snippet}")

    body = (
        f"User request:\n{query}\n\n"
        f"Retrieved documents:\n" + "\n\n".join(snippets) +
        "\n\nAre these documents relevant enough to help answer the request?"
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": body},
    ]