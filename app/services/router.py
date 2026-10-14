"""Routes an Intent to the correct pipeline (Registry pattern)."""
from __future__ import annotations

import logging

from app.domain.enums import Intent
from app.pipelines.base import Pipeline

logger = logging.getLogger(__name__)


class Router:
    """Extensible registry mapping Intent → Pipeline."""

    def __init__(self) -> None:
        self._registry: dict[Intent, Pipeline] = {}

    def register(self, intent: Intent, pipeline: Pipeline) -> "Router":
        self._registry[intent] = pipeline
        return self

    def route(self, intent: Intent) -> Pipeline:
        if intent not in self._registry:
            raise ValueError(f"No pipeline registered for intent: {intent}")
        logger.debug("Routing intent %s → %s", intent, type(self._registry[intent]).__name__)
        return self._registry[intent]