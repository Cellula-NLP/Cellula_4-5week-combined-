"""Prompt for the relevance evaluator."""
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
  • Same language + same domain                → relevant
  • Same language + adjacent domain            → relevant
  • Different language                          → not_relevant
  • Completely unrelated topic                  → not_relevant
  • Empty or placeholder documents              → not_relevant

SPECIAL RULE — user-provided solutions:
  If a document's `source` is "user_feedback" AND its `original_query`
  field is present, the document was provided by the user themselves as
  a solution for a past request. If the current request is the same or
  very similar to `original_query`, the document IS relevant — even if
  it is short or its language metadata looks incomplete.
  In that case, return {"verdict": "relevant", "score": 0.9}.

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
        original = d.metadata.get("original_query", "")
        snippet = d.content[:900]

        header = f"[{i}] source={src} · language={lang} · id={tid}"
        if original:
            header += f" · original_query={original!r}"

        snippets.append(f"{header}\n{snippet}")

    body = (
        f"User request:\n{query}\n\n"
        f"Retrieved documents:\n" + "\n\n".join(snippets) +
        "\n\nAre these documents relevant enough to help answer the request?"
    )

    return [
        Message(role="system", content=_SYSTEM),
        Message(role="user", content=body),
    ]