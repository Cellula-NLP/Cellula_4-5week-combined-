"""Orchestrates Tavily search, LLM extraction, and Chroma ingestion."""
from __future__ import annotations
import json
import logging
import re

from app.domain.ports import WebSearchProvider, LLMProvider
from app.prompts.extractor import build_extractor_prompt
from app.services.ingestion import IngestionService

logger = logging.getLogger(__name__)


class WebSearchService:
    def __init__(
            self,
            search_client: WebSearchProvider,
            llm: LLMProvider,
            ingestion: IngestionService
    ) -> None:
        self._search = search_client
        self._llm = llm
        self._ingestion = ingestion

    async def search_and_learn(self, query: str, session_id: str) -> tuple[str, str]:
        """Searches SO, extracts code, ingests it, and returns (code, language)."""
        logger.info("Searching Stack Overflow for: %s", query)
        results = await self._search.search(query)
        if not results:
            return "", "unknown"

        # Combine top 2 results to give the extractor more context
        combined_text = "\n\n---\n\n".join([r.content for r in results[:2]])

        try:
            messages = build_extractor_prompt(combined_text)
            raw_json = await self._llm.complete(messages, json_mode=True, temperature=0.0)
            data = self._parse_json(raw_json)

            code = data.get("code", "").strip()
            language = data.get("language", "unknown").strip()

            if len(code) < 20:
                logger.info("No substantial code found in search results.")
                return "", "unknown"

            logger.info("Extracted %s code (%d chars). Ingesting to Chroma.", language, len(code))

            # Ingest into Chroma so the next query hits the DB directly (Learning Loop)
            self._ingestion.ingest_solution(
                session_id=session_id,
                solution_text=code,
                original_query=query,
                language=language
            )
            return code, language

        except Exception as exc:
            logger.error("Failed to extract code from search: %s", exc)
            return "", "unknown"

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
            return {}