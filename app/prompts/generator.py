"""Prompt for code generation with retrieved context (RAG)."""
from __future__ import annotations

from app.domain.enums import MessageRole
from app.domain.models import Message, RetrievedDoc

_SYSTEM = """You are a senior software engineer who writes production-quality code.

OUTPUT LANGUAGE — CRITICAL:
- The code you output MUST be written in the language the user asked for.
- If the user says "write COBOL", output COBOL — not Python that wraps COBOL.
- If the user says "in Swift", output Swift — not Python that returns Swift.
- If the user says "convert to Rust", output Rust — not Python.
- NEVER wrap the target language inside a Python string unless the user
  explicitly asked for "a Python function that generates <other language>".
- If the target language cannot actually run in the sandbox, output the
  code anyway — the sandbox is not a limitation on what you write.

CODE GENERATION RULES:
- Produce complete source code inside a single fenced code block.
- ALWAYS tag the fence with the target language, e.g. ```python, ```swift, ```cobol.
- Include helpful comments and docstrings.
- Handle errors and edge cases explicitly.
- Follow best practices for the target language/framework.
- After the code block, add a short "Explanation" section (2-5 bullets).
- If retrieved context is provided, use it as reference but do NOT copy blindly.
- If you are unsure, say so rather than hallucinating APIs.

RUNNABLE DEMO — only for Python / JavaScript / Bash:
- If the target language is Python, JavaScript, or Bash, end the code block
  with a self-contained demo that runs 2-3 examples and prints results.
- For other languages (Swift, Rust, COBOL, Java, Go, etc.), do NOT add a
  demo block — just output clean idiomatic code for that language.
"""


def build_generator_prompt(
    query: str,
    docs: list[RetrievedDoc],
    history: list[Message],
    language: str | None = None,
) -> list[Message]:
    """Build the generator prompt with optional retrieved context."""
    messages: list[Message] = [Message(role=MessageRole.SYSTEM, content=_SYSTEM)]

    for m in history[-6:]:
        if m.role in (MessageRole.USER, MessageRole.ASSISTANT):
            messages.append(m)

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
            "Use these as guidance but write your own complete solution "
            "IN THE LANGUAGE THE USER REQUESTED."
        )

    lang_hint = f"\n**Output language: {language}**" if language else ""
    body = f"{query}{lang_hint}{context_block}"
    messages.append(Message(role=MessageRole.USER, content=body))
    return messages