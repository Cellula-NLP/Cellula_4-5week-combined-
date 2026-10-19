"""Tavily API client for searching Stack Overflow."""
from __future__ import annotations

import logging
import httpx

from app.domain.ports import WebSearchProvider
from app.domain.models import WebSearchResult

logger = logging.getLogger(__name__)


class TavilySearchClient(WebSearchProvider):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._url = "https://api.tavily.com/search"

    async def search(self, query: str, max_results: int = 3) -> list[WebSearchResult]:
        if not self._api_key:
            logger.warning("Tavily API key missing. Web search skipped.")
            return []

        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "advanced",
            "include_domains": ["stackoverflow.com"],
            "max_results": max_results,
            "include_raw_content": True,
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(self._url, json=payload)
                resp.raise_for_status()
                data = resp.json()

                results = []
                for r in data.get("results", []):
                    results.append(WebSearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        content=r.get("raw_content") or r.get("content", "")
                    ))
                return results
        except Exception as exc:
            logger.error("Tavily search failed: %s", exc)
            return []