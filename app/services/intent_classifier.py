"""LLM-based intent classifier with heuristic fallback.

Classifies every incoming query into exactly one of:
  EXPLAIN_CODE | GENERATE_CODE | CLARIFY | OUT_OF_SCOPE
"""
from __future__ import annotations

import json
import logging
import re

from app.core.exceptions import ClassificationError
from app.domain.enums import Intent
from app.domain.models import Classification, Message
from app.domain.ports import LLMProvider
from app.prompts.classifier import build_classifier_prompt

logger = logging.getLogger(__name__)

# Cheap heuristic signals (used as pre-filter and fallback).
_GEN_VERBS = (
    "write", "generate", "create", "build", "implement",
    "make a", "develop", "code a", "give me a",
)
_EXPLAIN_VERBS = (
    "explain", "what does", "what is", "why is", "why does", "why i",
    "why am i", "why do i", "why my", "why this", "describe",
    "walk me through", "line by line", "how does", "how do",
    "error in", "errors in", "bug in", "wrong with", "issue with",
    "what's wrong", "whats wrong", "help me understand",
)


class IntentClassifier:
    """Classifies a user query into an Intent."""

    def __init__(self, llm: LLMProvider, min_confidence: float = 0.55) -> None:
        self._llm = llm
        self._min_conf = min_confidence

    async def classify(
        self,
        query: str,
        code_context: str | None = None,
    ) -> Classification:
        # If the heuristic is very confident, skip the LLM call entirely.
        heuristic = self._heuristic(query, code_context)
        if heuristic is not None and heuristic.confidence >= 0.85:
            return heuristic

        # 2) LLM structured output (authoritative).
        try:
            messages = build_classifier_prompt(query, code_context)
            raw = await self._llm.complete(
                messages, json_mode=True, temperature=0.0, max_tokens=250,
            )
            data = self._parse_json(raw)
            intent = Intent(data["intent"])
            confidence = float(data.get("confidence", 0.5))
            language = data.get("language")
            reasoning = data.get("reasoning", "")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("LLM classifier failed (%s). Using heuristic.", exc)
            if heuristic is not None:
                return heuristic
            return Classification(Intent.CLARIFY, 0.0, reasoning="classifier failure")

        # 3) Confidence guard.
        if confidence < self._min_conf:
            logger.info("Low confidence (%.2f) → CLARIFY", confidence)
            return Classification(
                Intent.CLARIFY, confidence,
                language=language, reasoning=reasoning or "low confidence",
            )

        return Classification(
            intent=intent, confidence=confidence,
            language=language, extracted_code=code_context, reasoning=reasoning,
        )

    # ── Internals ─────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Extract the first JSON object from a possibly-noisy LLM response."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
            raw = re.sub(r"\n```$", "", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    def _heuristic(self, query: str, code: str | None) -> Classification | None:
        ql = query.lower()

        has_gen = any(v in ql for v in _GEN_VERBS)
        has_exp = any(v in ql for v in _EXPLAIN_VERBS)

        # Code pasted + no generation verb → almost certainly EXPLAIN.
        if code and not has_gen:
            return Classification(Intent.EXPLAIN_CODE, 0.85, extracted_code=code,
                                  reasoning="heuristic: code present, no gen verb")

        # Clear generation intent, no explanation ask.
        if has_gen and not has_exp:
            return Classification(Intent.GENERATE_CODE, 0.85,
                                  reasoning="heuristic: generation verb")

        # Clear explanation intent.
        if has_exp and not has_gen:
            return Classification(Intent.EXPLAIN_CODE, 0.85, extracted_code=code,
                                  reasoning="heuristic: explanation verb")

        return None  # ambiguous → let the LLM decide