"""LLM-based relevance evaluator.

Task requirement (PDF page 9): ask an LLM whether the retrieved
context is sufficiently relevant. If not, the pipeline must not
hallucinate — it should ask the user for the correct solution.

Two shortcuts:
  • If any doc is a user-provided solution whose original_query matches
    the current request, trust it and return RELEVANT without an LLM call.
  • If the top bi-encoder score is below a hard cutoff, return
    NOT_RELEVANT immediately without an LLM call. The cutoff is set at
    the retriever's ``min_score`` so it fires only when the candidate
    set consists of low-score feedback docs that slipped through the
    similarity gate (see Retriever._fetch_matching_feedback_docs).
"""
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
    # Below this bi-encoder score, skip the LLM call. Matches the
    # retriever's min_score so we never call the evaluator when the
    # candidate set is uniformly weak.
    _HARD_CUTOFF = 0.25

    def __init__(
        self,
        llm: LLMProvider,
        threshold: float = 0.35,
    ) -> None:
        self._llm = llm
        self._threshold = threshold

    async def evaluate(self, query: str, docs: list[RetrievedDoc]) -> Verdict:
        if not docs:
            return Verdict(Relevance.NOT_RELEVANT, 0.0, "no documents retrieved")

        # Shortcut 1 — user-feedback doc matching the current query.
        for d in docs:
            if d.metadata.get("source") != "user_feedback":
                continue
            original = (d.metadata.get("original_query") or "").strip()
            if original and queries_match(query, original):
                logger.info(
                    "User-feedback doc matches the current query — "
                    "short-circuiting relevance check."
                )
                return Verdict(
                    Relevance.RELEVANT, 1.0,
                    "user-provided solution for matching query",
                )

        # Shortcut 2 — obviously weak retrieval.
        top_score = max(d.score for d in docs)
        if top_score < self._HARD_CUTOFF:
            logger.info(
                "Top score %.3f < hard cutoff %.2f — skipping evaluator LLM call.",
                top_score, self._HARD_CUTOFF,
            )
            return Verdict(
                Relevance.NOT_RELEVANT, top_score,
                f"top score {top_score:.2f} below hard cutoff",
            )

        try:
            messages = build_relevance_prompt(query, docs)
            raw = await self._llm.complete(
                messages, json_mode=True, temperature=0.0, max_tokens=200,
            )
            data = self._parse_json(raw)
            score = float(data.get("score", 0.0))
            verdict_str = str(data.get("verdict", "not_relevant")).lower().strip()
            reason = str(data.get("reason", "")).strip()

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