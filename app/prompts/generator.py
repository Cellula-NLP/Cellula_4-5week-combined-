"""Prompt for code generation with retrieved context (RAG)."""
from __future__ import annotations

from app.domain.enums import MessageRole
from app.domain.models import Message, RetrievedDoc

_SYSTEM = """You are a senior software engineer who writes production-quality code.

Rules:
- Produce complete, runnable source code inside a single fenced code block.
- ALWAYS tag the fence with the target language, e.g. ```python or ```javascript.
  Never use a bare ``` fence.
- Include helpful comments and docstrings.
- Handle errors and edge cases explicitly.
- Follow best practices for the target language/framework.
- After the code block, add a short "Explanation" section (2-5 bullets).
- If retrieved context is provided, use it as reference but do NOT copy blindly.
- If you are unsure, say so rather than hallucinating APIs.

IMPORTANT — Always include a runnable demo:
- For Python: end the code block with a

    if __name__ == "__main__":
        ...

  section that calls the function with 2-3 representative examples
  (including at least one edge case) and prints the results with `print()`.
- For JavaScript: append a short demo that calls the function and uses
  `console.log(...)`.
- The demo must be self-contained: no external files, no network, no
  user input. Sample data should be hardcoded.
- This lets the user click "Run" and immediately see output.
"""


def build_generator_prompt(
    query: str,
    docs: list[RetrievedDoc],
    history: list[Message],
    language: str | None = None,
) -> list[Message]:
    """Build the generator prompt with optional retrieved context."""
    messages: list[Message] = [Message(role=MessageRole.SYSTEM, content=_SYSTEM)]

    # Short conversation memory — last 6 user/assistant turns.
    for m in history[-6:]:
        if m.role in (MessageRole.USER, MessageRole.ASSISTANT):
            messages.append(m)

    # Retrieved context (RAG).
    context_block = ""
    if docs:
        snippets: list[str] = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            tid = d.metadata.get("task_id", d.id)
            lang = d.metadata.get("language", "python")
            snippets.append(
                f"[{i}] ({src} · {tid} · {lang})\n```{lang}\n{d.content}\n```"
            )
        context_block = (
            "\n\n=== Retrieved reference implementations ===\n"
            + "\n\n".join(snippets)
            + "\n=== End of references ===\n"
            "Use these as guidance but write your own complete solution."
        )

    lang_hint = f"\nPreferred language: {language}" if language else ""
    body = f"{query}{lang_hint}{context_block}"
    messages.append(Message(role=MessageRole.USER, content=body))
    return messages