from __future__ import annotations
import json
import logging
import re
from app.domain.enums import Relevance
from app.domain.models import RetrievedDoc, Verdict
from app.domain.ports import LLMProvider
from app.prompts.relevance import build_relevance_prompt
from app.services.query_similarity import queries_match

logger = logging.getLogger(__name__)

class RelevanceEvaluator:
    _HARD_CUTOFF = 0.25

    def __init__(self, llm: LLMProvider, threshold: float = 0.35) -> None:
        self._llm = llm
        self._threshold = threshold

    async def evaluate(self, query: str, docs: list[RetrievedDoc]) -> Verdict:
        if not docs:
            return Verdict(Relevance.NOT_RELEVANT, 0.0, "no documents retrieved")

        for d in docs:
            if d.metadata.get("source") != "user_feedback" and d.metadata.get("source") != "tavily_so":
                continue
            original = (d.metadata.get("original_query") or "").strip()
            if original and queries_match(query, original):
                return Verdict(Relevance.RELEVANT, 1.0, "matching taught solution")

        top_score = max(d.score for d in docs)
        if top_score < self._HARD_CUTOFF:
            return Verdict(Relevance.NOT_RELEVANT, top_score, "below hard cutoff")

        try:
            messages = build_relevance_prompt(query, docs)
            raw = await self._llm.complete(messages, json_mode=True, temperature=0.0)
            data = self._parse_json(raw)
            score = float(data.get("score", 0.0))
            verdict_str = str(data.get("verdict", "not_relevant")).lower().strip()

            if verdict_str == "relevant" and score >= self._threshold:
                return Verdict(Relevance.RELEVANT, score, data.get("reason", ""))
            return Verdict(Relevance.NOT_RELEVANT, score, data.get("reason", ""))
        except Exception as exc:
            logger.warning("Evaluator failed: %s", exc)
            return Verdict(Relevance.NOT_RELEVANT, 0.0, "evaluator failure")

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