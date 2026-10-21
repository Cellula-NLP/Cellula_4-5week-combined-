"""Prompt to extract ONLY valid code from messy Stack Overflow HTML text."""
from __future__ import annotations

_SYSTEM = """You are a strict code extraction tool.
The user will provide the raw text of a Stack Overflow answer.
Your job is to extract the most complete, runnable code snippet from the text.

Respond with STRICT JSON only (no markdown, no prose):
{
  "language": "python" | "javascript" | "cobol" | "unknown",
  "code": "The exact code block extracted, without markdown fences."
}

Rules:
- If no valid code exists, return {"language": "unknown", "code": ""}.
- Do NOT include explanations in the "code" field.
- Strip any markdown ``` backticks.
"""

def build_extractor_prompt(raw_web_text: str) -> list[dict]:
    # Truncate to 6000 chars to avoid token limits
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Extract code from this:\n\n{raw_web_text[:6000]}"}
    ]