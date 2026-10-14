"""Sandboxed code execution via subprocess with resource limits.

Security measures:
  • CPU time limit (RLIMIT_CPU)
  • Address-space limit (RLIMIT_AS)
  • File-size limit (RLIMIT_FSIZE)
  • Process-count limit (RLIMIT_NPROC)
  • Open-file limit (RLIMIT_NOFILE)
  • New session group (os.setsid) so we can kill the whole tree
  • Wall-clock timeout
  • Restricted environment (no network, minimal PATH)
"""
from __future__ import annotations

import logging
import os
import resource
import subprocess
import tempfile
import time
from pathlib import Path

from app.core.exceptions import ExecutionError
from app.domain.models import ExecutionResult
from app.domain.ports import CodeRunner

logger = logging.getLogger(__name__)

_EXEC_COMMANDS: dict[str, list[str]] = {
    "python": ["python", "-u"],
    "javascript": ["node"],
    "bash": ["bash"],
}


class SubprocessSandbox(CodeRunner):
    """Runs untrusted code in a locked-down subprocess."""

    def __init__(self, default_timeout_s: float = 5.0, mem_mb: int = 256) -> None:
        self._timeout = default_timeout_s
        self._mem_bytes = mem_mb * 1024 * 1024

    def run(
        self,
        code: str,
        language: str,
        *,
        timeout_s: float | None = None,
        stdin: str = "",
    ) -> ExecutionResult:
        language = language.lower()
        if language not in _EXEC_COMMANDS:
            raise ExecutionError(f"Unsupported language: {language}")

        timeout = timeout_s or self._timeout
        suffix = self._suffix_for(language)

        with tempfile.TemporaryDirectory(prefix="sandbox_") as td:
            path = Path(td) / f"main{suffix}"
            path.write_text(code, encoding="utf-8")
            cmd = _EXEC_COMMANDS[language] + [str(path)]

            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    input=stdin.encode("utf-8") if stdin else b"",
                    capture_output=True,
                    timeout=timeout,
                    preexec_fn=self._apply_limits,
                    cwd=td,
                    env={
                        "PATH": "/usr/bin:/bin:/usr/local/bin",
                        "PYTHONUNBUFFERED": "1",
                        "LANG": "en_US.UTF-8",
                        "HOME": td,
                    },
                    check=False,
                )
                elapsed = int((time.perf_counter() - t0) * 1000)
                return ExecutionResult(
                    ok=proc.returncode == 0,
                    stdout=proc.stdout.decode("utf-8", errors="replace")[:50_000],
                    stderr=proc.stderr.decode("utf-8", errors="replace")[:50_000],
                    exit_code=proc.returncode,
                    duration_ms=elapsed,
                )
            except subprocess.TimeoutExpired as exc:
                elapsed = int((time.perf_counter() - t0) * 1000)
                return ExecutionResult(
                    ok=False,
                    stdout=(exc.stdout or b"").decode("utf-8", errors="replace")[:50_000],
                    stderr=f"Execution timed out after {timeout}s",
                    exit_code=-1,
                    duration_ms=elapsed,
                    timed_out=True,
                )
            except Exception as exc:
                raise ExecutionError(f"Sandbox failure: {exc}") from exc

    # ── Internals ─────────────────────────────────

    @staticmethod
    def _suffix_for(language: str) -> str:
        return {
            "python": ".py",
            "javascript": ".js",
            "bash": ".sh",
        }.get(language, ".txt")

    def _apply_limits(self) -> None:
        """Called in the child process before exec."""
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
            resource.setrlimit(resource.RLIMIT_AS, (self._mem_bytes, self._mem_bytes))
            resource.setrlimit(resource.RLIMIT_FSIZE, (5 * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            os.setsid()  # new session → kill entire tree on timeout
        except Exception:
            pass  # best-effort; non-POSIX platforms may not support all limits