"""Service wrapper around the CodeRunner port."""
from __future__ import annotations

import logging

from app.domain.models import ExecutionResult
from app.domain.ports import CodeRunner

logger = logging.getLogger(__name__)


class CodeExecutor:
    def __init__(self, runner: CodeRunner) -> None:
        self._runner = runner

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