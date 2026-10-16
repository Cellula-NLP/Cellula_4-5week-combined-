"""Service wrapper around the CodeRunner port."""
from __future__ import annotations

import logging

from app.domain.models import ExecutionResult
from app.domain.ports import CodeRunner

logger = logging.getLogger(__name__)


class CodeExecutor:
    def __init__(self, runner: CodeRunner) -> None:
        self._runner = runner

    @property
    def available_languages(self) -> frozenset[str]:
        """Languages the underlying runner can actually execute.

        Used by the UI to gate Run buttons — a code block in a language
        whose interpreter isn't installed must not show a Run button.
        Falls back to an empty set if the runner doesn't advertise any.
        """
        return getattr(self._runner, "available_languages", frozenset())

    def execute(
        self,
        code: str,
        language: str,
        *,
        timeout_s: float | None = None,
        stdin: str = "",
    ) -> ExecutionResult:
        logger.info("Executing %s code (%d chars) …", language, len(code))
        result = self._runner.run(code, language, timeout_s=timeout_s, stdin=stdin)
        logger.info("Execution finished: ok=%s, exit=%s, %d ms",
                    result.ok, result.exit_code, result.duration_ms)
        return result