"""Prompt for the relevance evaluator.

Task requirement (PDF page 9): after retrieval, the documents must be
passed to an LLM that answers:
    "Is the retrieved context sufficiently relevant to answer the user's request?"
Possible outputs: Relevant / Not Relevant.

The evaluator must be strict enough to prevent hallucination but fair
enough to accept related content (e.g. "reverse a string" is relevant
when HumanEval returns other Python string helpers).
"""
from __future__ import annotations

from app.domain.models import Message, RetrievedDoc

_SYSTEM = """You evaluate whether retrieved coding documents are relevant \
enough to help answer a user's programming request.

Respond with STRICT JSON only (no markdown, no prose):
{
  "verdict": "relevant" | "not_relevant",
  "score": 0.0-1.0,
  "reason": "one short sentence"
}

Definition of "relevant":
  The documents contain code, patterns, or concepts that share the same
  programming language AND the same general problem domain as the request.
  They do NOT need to be an exact match.

Definition of "not_relevant":
  Using the documents would clearly mislead the model — different
  language, different domain, or completely unrelated topic.

Decision guide:
  • Same language + same domain (e.g. both Python string operations)  → relevant
  • Same language + adjacent domain (e.g. Python list ops for a Python
    string request)                                                    → relevant
  • Different language                                                → not_relevant
  • Completely unrelated topic (e.g. CSV parsing for a linked list)   → not_relevant
  • Empty or placeholder documents                                    → not_relevant

Scoring:
  • 0.9-1.0  : documents directly address the request
  • 0.6-0.8  : documents are in the same domain and would help
  • 0.4-0.5  : documents are tangentially related
  • 0.0-0.3  : documents are unrelated or misleading

When uncertain between 0.5 and 0.6, prefer "relevant" — the generator
will still write its own solution rather than copy the references.
"""


def build_relevance_prompt(query: str, docs: list[RetrievedDoc]) -> list[Message]:
    """Build the evaluator prompt for the given query and retrieved docs."""
    snippets: list[str] = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        tid = d.metadata.get("task_id", d.id)
        lang = d.metadata.get("language", "unknown")
        # Truncate each doc so the prompt stays small and fast.
        snippet = d.content[:900]
        snippets.append(
            f"[{i}] source={src} · language={lang} · id={tid}\n{snippet}"
        )

    body = (
        f"User request:\n{query}\n\n"
        f"Retrieved documents:\n" + "\n\n".join(snippets) +
        "\n\nAre these documents relevant enough to help answer the request?"
    )

    return [
        Message(role="system", content=_SYSTEM),
        Message(role="user", content=body),
    ]