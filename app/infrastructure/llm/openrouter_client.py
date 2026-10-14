"""OpenRouter LLM client (OpenAI-compatible API).

Model strategy:
  1. Primary model   (settings.llm_model)
  2. Fallback model  (settings.llm_fallback_model)
  3. Free Models Router ("openrouter/free") — automatically selects any
     available free model. This is the ultimate safety net so the app
     never breaks when individual free model IDs are removed/changed.

Handles:
  • 404 (model not found)      → try next model
  • 429 (rate limit)           → tenacity retry + try next model
  • 5xx (server error)         → retry + try next model
  • Timeouts / connect errors  → retry + try next model
  • Malformed JSON responses   → LLMError
  • Empty content              → treated as transient, try next model
  • Streaming SSE parsing      → robust against partial chunks

Caching (Win C):
  • Non-streamed completions are cached by a hash of (model chain,
    messages, temperature, JSON_mode, max_tokens).
  • Bounded LRU-style cache prevents unbounded memory growth.
  • Thread-safe (Streamlit runs scripts in worker threads).
  • Streaming responses are NOT cached (they'd need full buffering).
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import AsyncIterator, Sequence

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.config import Settings
from app.core.exceptions import LLMError, RateLimitError, TimeoutError
from app.domain.models import Message
from app.domain.ports import LLMProvider

logger = logging.getLogger(__name__)

# Exceptions that justify a retry at the tenacity level.
_RETRYABLE = (httpx.TimeoutException, httpx.ConnectError, RateLimitError)

# The universal free-model fallback. OpenRouter routes to whatever free
# model is available at request time.
_FREE_ROUTER = "openrouter/free"

# Maximum number of cached completions. Small enough to never be an issue.
_DEFAULT_CACHE_SIZE = 256


class OpenRouterClient(LLMProvider):
    """Async OpenRouter client with retry, model fallback, and JSON mode."""

    def __init__(
        self,
        settings: Settings,
        *,
        cache_size: int = _DEFAULT_CACHE_SIZE,
        enable_cache: bool = True,
    ) -> None:
        self._settings = settings
        self._enable_cache = enable_cache
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        self._cache_size = max(0, cache_size)
        self._cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0

        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url,
            timeout=httpx.Timeout(settings.llm_timeout_s),
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/ai-coding-assistant",
                "X-Title": "AI Coding Assistant",
            },
        )

    # ── Public API ────────────────────────────────

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Non-streamed completion with cache + fallback chain."""
        cache_key = self._make_cache_key(
            messages, json_mode, temperature, max_tokens,
        )
        if cache_key is not None:
            cached = self._cache_get(cache_key)
            if cached is not None:
                logger.debug("LLM cache hit (%d chars).", len(cached))
                return cached

        last_exc: Exception | None = None
        for model in self._model_chain():
            try:
                result = await self._complete_once(
                    model, messages, json_mode, temperature, max_tokens,
                )
                if cache_key is not None:
                    self._cache_put(cache_key, result)
                return result
            except (httpx.HTTPStatusError, httpx.TimeoutException, LLMError) as exc:
                logger.warning("Model %s failed: %s. Trying next…", model, exc)
                last_exc = exc

        raise LLMError(f"All models failed. Last error: {last_exc}") from last_exc

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens via SSE with the same fallback chain (no caching)."""
        last_exc: Exception | None = None
        for model in self._model_chain():
            try:
                async for token in self._stream_once(
                    model, messages, temperature, max_tokens,
                ):
                    yield token
                return
            except (
                httpx.HTTPStatusError,
                httpx.ConnectError,
                httpx.TimeoutException,
                LLMError,
            ) as exc:
                logger.warning("Stream model %s failed: %s. Trying next…", model, exc)
                last_exc = exc
        raise LLMError(
            f"All streaming models failed. Last error: {last_exc}"
        ) from last_exc

    # ── Cache API (exposed for debugging/monitoring) ──

    @property
    def cache_stats(self) -> dict:
        with self._cache_lock:
            total = self._cache_hits + self._cache_misses
            return {
                "size": len(self._cache),
                "capacity": self._cache_size,
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": (self._cache_hits / total) if total else 0.0,
            }

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    # ── Internals ─────────────────────────────────

    def _model_chain(self) -> list[str]:
        """Ordered, de-duplicated list of models to try."""
        candidates = [
            self._settings.llm_model,
            self._settings.llm_fallback_model,
            _FREE_ROUTER,
        ]
        seen: set[str] = set()
        chain: list[str] = []
        for m in candidates:
            if m and m not in seen:
                seen.add(m)
                chain.append(m)
        return chain

    # ── Cache helpers ─────────────────────────────

    def _make_cache_key(
        self,
        messages: Sequence[Message],
        json_mode: bool,
        temperature: float | None,
        max_tokens: int | None,
    ) -> str | None:
        """Deterministic hash of all inputs that affect the response.

        Returns None if caching is disabled.
        """
        if not self._enable_cache or self._cache_size == 0:
            return None
        try:
            payload = {
                "chain": self._model_chain(),
                "messages": [
                    {"role": m.role.value, "content": m.content} for m in messages
                ],
                "json_mode": json_mode,
                "temperature": (
                    temperature if temperature is not None
                    else self._settings.llm_temperature
                ),
                "max_tokens": max_tokens,
            }
            encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        except (TypeError, ValueError) as exc:
            logger.warning("Could not build cache key: %s", exc)
            return None

    def _cache_get(self, key: str) -> str | None:
        with self._cache_lock:
            if key in self._cache:
                # Move to end → marks as recently used.
                self._cache.move_to_end(key)
                self._cache_hits += 1
                return self._cache[key]
            self._cache_misses += 1
            return None

    def _cache_put(self, key: str, value: str) -> None:
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
                return
            self._cache[key] = value
            # Evict least-recently-used entry if over capacity.
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    # ── Network helpers ───────────────────────────

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        reraise=True,
    )
    async def _complete_once(
        self,
        model: str,
        messages: Sequence[Message],
        json_mode: bool,
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        body = self._build_body(model, messages, temperature, max_tokens)
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        resp = await self._client.post("/chat/completions", json=body)

        if resp.status_code == 429:
            raise RateLimitError(f"Rate limit hit for {model}")
        if resp.status_code >= 500:
            raise LLMError(f"Server error {resp.status_code} for {model}")
        resp.raise_for_status()

        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(
                f"Unexpected response shape for {model}: {data}"
            ) from exc

        # Some OpenRouter free-tier models return None content alongside a
        # valid finish_reason. Treat it as a transient failure so the
        # fallback chain can try the next model.
        if content is None or not content.strip():
            raise LLMError(f"Empty content from {model}")
        return content

    async def _stream_once(
        self,
        model: str,
        messages: Sequence[Message],
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncIterator[str]:
        body = self._build_body(model, messages, temperature, max_tokens)
        body["stream"] = True

        async with self._client.stream("POST", "/chat/completions", json=body) as resp:
            if resp.status_code == 429:
                raise RateLimitError(f"Rate limit hit for {model}")
            if resp.status_code >= 500:
                raise LLMError(f"Server error {resp.status_code} for {model}")
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    return
                try:
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    # Ignore malformed SSE chunks; keep streaming.
                    continue

    def _build_body(
        self,
        model: str,
        messages: Sequence[Message],
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict:
        return {
            "model": model,
            "messages": [
                {"role": m.role.value, "content": m.content} for m in messages
            ],
            "temperature": (
                temperature if temperature is not None
                else self._settings.llm_temperature
            ),
            "max_tokens": max_tokens,
        }

    async def aclose(self) -> None:
        await self._client.aclose()