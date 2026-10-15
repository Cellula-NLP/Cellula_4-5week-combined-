"""LLM-based relevance evaluator.

Task requirement (PDF page 9): ask an LLM whether the retrieved
context is sufficiently relevant. If not, the pipeline must not
hallucinate — it should ask the user for the correct solution.

Default on any evaluator failure: NOT_RELEVANT (fail-safe, prevents
hallucination per the task).
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
        # Empty retrieval → NOT_RELEVANT immediately (no LLM call needed).
        if not docs:
            return Verdict(Relevance.NOT_RELEVANT, 0.0, "no documents retrieved")

        try:
            messages = build_relevance_prompt(query, docs)
            raw = await self._llm.complete(
                messages, json_mode=True, temperature=0.0, max_tokens=200,
            )
            data = self._parse_json(raw)

            score = float(data.get("score", 0.0))
            verdict_str = str(data.get("verdict", "not_relevant")).lower().strip()
            reason = str(data.get("reason", "")).strip()

            # Trust the verdict, but require a minimum score too.
            if verdict_str == "relevant" and score >= self._threshold:
                return Verdict(Relevance.RELEVANT, score, reason)

            return Verdict(Relevance.NOT_RELEVANT, score, reason)

        except Exception as exc:
            logger.warning(
                "Relevance evaluator failed: %s. Defaulting NOT_RELEVANT.", exc,
            )
            return Verdict(
                Relevance.NOT_RELEVANT, 0.0, f"evaluator failure: {exc}",
            )

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