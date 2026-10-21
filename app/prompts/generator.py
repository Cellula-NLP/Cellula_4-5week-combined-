"""Prompt for code generation with retrieved context (RAG)."""
from __future__ import annotations
from app.domain.models import RetrievedDoc

_SYSTEM = """You are a senior software engineer who writes production-quality code.
OUTPUT LANGUAGE — CRITICAL: The code MUST be in the language requested.
CODE GENERATION RULES:
- Produce complete source code inside a single fenced code block.
- ALWAYS tag the fence with the language.
- Include helpful comments.
- After the code block, add a short "Explanation" section.
"""

def build_generator_prompt(
    query: str,
    docs: list[RetrievedDoc],
    history: list[dict],
    language: str | None = None,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": _SYSTEM}]

    for m in history[-6:]:
        messages.append(m)

    context_block = ""
    if docs:
        snippets = [f"[{i}] ({d.metadata.get('source', 'unknown')})\n```\n{d.content}\n```" for i, d in enumerate(docs, 1)]
        context_block = "\n\n=== Retrieved reference implementations ===\n" + "\n\n".join(snippets) + "\n=== End of references ===\n"

    lang_hint = f"\n**Output language: {language}**" if language else ""
    body = f"{query}{lang_hint}{context_block}"
    messages.append({"role": "user", "content": body})
    return messages