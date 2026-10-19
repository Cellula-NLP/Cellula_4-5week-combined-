from __future__ import annotations
from app.domain.enums import MessageRole
from app.domain.models import Message

_SYSTEM = """You are an expert code extractor. 
The user will provide raw text scraped from Stack Overflow.
Your job is to find the most complete, valid code snippet in the text and return it.

Respond with STRICT JSON only (no markdown, no prose):
{
  "language": "python" | "javascript" | "swift" | "cobol" | "unknown",
  "code": "The exact code block extracted, without markdown fences."
}

Rules:
- Ignore all HTML tags, Stack Overflow formatting, and conversational text.
- If you see multiple snippets, pick the longest, most complete one.
- Do NOT include explanations in the "code" field. Just the raw code.
- If absolutely no code exists, return {"language": "unknown", "code": ""}.
"""

def build_extractor_prompt(raw_web_text: str) -> list[Message]:
    # We increase the character limit slightly to 7000 to ensure we don't miss context
    return [
        Message(role=MessageRole.SYSTEM, content=_SYSTEM),
        Message(role=MessageRole.USER, content=f"Extract the best code snippet from this Stack Overflow page:\n\n{raw_web_text[:7000]}")
    ]