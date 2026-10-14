"""LLM-based relevance evaluator.

Answers: 'Is the retrieved context sufficiently relevant to answer the request?'
Default on any failure: NOT_RELEVANT (never hallucinate).
"""
from __future__ import annotations

import json
import logging
import re

from app.domain.enums import Relevance
from app.domain.models import RetrievedDoc, Verdict
from app.domain.ports import LLMProvider
from app.prompts.relevance import build_relevance_prompt

logger = logging.getLogger(__name__)


class RelevanceEvaluator:
    def __init__(
        self,
        llm: LLMProvider,
        threshold: float = 0.5,
    ) -> None:
        self._llm = llm
        self._threshold = threshold

    async def evaluate(self, query: str, docs: list[RetrievedDoc]) -> Verdict:
        # Empty retrieval → immediately NOT_RELEVANT.
        if not docs:
            return Verdict(Relevance.NOT_RELEVANT, 0.0, "no documents retrieved")

        try:
            messages = build_relevance_prompt(query, docs)
            raw = await self._llm.complete(
                messages, json_mode=True, temperature=0.0, max_tokens=200,
            )
            data = self._parse_json(raw)
            score = float(data.get("score", 0.0))
            verdict_str = str(data.get("verdict", "not_relevant")).lower()
            reason = data.get("reason", "")

            if verdict_str == "relevant" and score >= self._threshold:
                return Verdict(Relevance.RELEVANT, score, reason)
            return Verdict(Relevance.NOT_RELEVANT, score, reason)

        except Exception as exc:
            logger.warning("Relevance evaluator failed: %s. Defaulting NOT_RELEVANT.", exc)
            return Verdict(Relevance.NOT_RELEVANT, 0.0, f"evaluator failure: {exc}")

    @staticmethod
    def _parse_json(raw: str) -> dict:
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