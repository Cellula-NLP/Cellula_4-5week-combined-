"""Prompt for the relevance evaluator."""
from __future__ import annotations

from app.domain.models import Message, RetrievedDoc

_SYSTEM = """You evaluate whether retrieved documents are relevant enough to answer a user's coding request.

Respond with STRICT JSON only:
{
  "verdict": "relevant" | "not_relevant",
  "score": 0.0-1.0,
  "reason": "one short sentence"
}

Be strict: if the documents only tangentially match or would lead to hallucination, return "not_relevant" with a low score.
"""


def build_relevance_prompt(query: str, docs: list[RetrievedDoc]) -> list[Message]:
    snippets = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        tid = d.metadata.get("task_id", d.id)
        snippets.append(f"[{i}] source={src} id={tid}\n{d.content[:800]}")

    user = (
        f"User request:\n{query}\n\n"
        f"Retrieved documents:\n" + "\n\n".join(snippets) +
        "\n\nAre these documents relevant enough to answer the request?"
    )
    return [
        Message(role="system", content=_SYSTEM),
        Message(role="user", content=user),
    ]