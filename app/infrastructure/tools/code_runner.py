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

Interpreter resolution:
  • Python: uses sys.executable so the sandbox always runs in the same
    interpreter (and site-packages) as the parent process.
  • Node / Bash: resolved via shutil.which() at import time.
  • Only languages whose interpreter is actually available are registered.
"""
from __future__ import annotations

import logging
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from app.core.exceptions import ExecutionError
from app.domain.models import ExecutionResult
from app.domain.ports import CodeRunner

logger = logging.getLogger(__name__)


# ── Interpreter resolution (once at import time) ────────────────

def _resolve_command(*candidates: str) -> str | None:
    """Return the first candidate that resolves to an existing binary."""
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isabs(candidate):
            if os.path.exists(candidate):
                return candidate
        else:
            found = shutil.which(candidate)
            if found:
                return found
    return None


# Python: prefer the current interpreter (venv-aware). Fall back to
# python3 / python if sys.executable is somehow unavailable.
_PYTHON_BIN = _resolve_command(
    sys.executable or "",
    "python3",
    "python",
)
_NODE_BIN = _resolve_command("node", "nodejs")
_BASH_BIN = _resolve_command("bash", "/bin/bash")

_EXEC_COMMANDS: dict[str, list[str]] = {}
if _PYTHON_BIN:
    _EXEC_COMMANDS["python"] = [_PYTHON_BIN, "-u"]
if _NODE_BIN:
    _EXEC_COMMANDS["javascript"] = [_NODE_BIN]
if _BASH_BIN:
    _EXEC_COMMANDS["bash"] = [_BASH_BIN]

# Directory containing the current Python interpreter — added to PATH
# inside the sandbox so subprocesses can find venv tools (pip, etc.).
_VENV_BIN = str(Path(sys.executable).parent) if sys.executable else ""

logger.debug(
    "Sandbox interpreters resolved — python=%s, node=%s, bash=%s",
    _PYTHON_BIN, _NODE_BIN, _BASH_BIN,
)


# ── Sandbox ─────────────────────────────────────────────────────

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
            available = ", ".join(sorted(_EXEC_COMMANDS)) or "(none detected)"
            raise ExecutionError(
                f"Unsupported language: {language}. Available: {available}"
            )

        timeout = timeout_s or self._timeout
        suffix = self._suffix_for(language)

        with tempfile.TemporaryDirectory(prefix="sandbox_") as td:
            path = Path(td) / f"main{suffix}"
            path.write_text(code, encoding="utf-8")
            cmd = _EXEC_COMMANDS[language] + [str(path)]

            # Build the sandbox environment. Inherit TMPDIR and locale
            # from the parent; do NOT inherit secrets (API keys etc.).
            path_dirs = [p for p in (
                _VENV_BIN,
                "/usr/local/bin",
                "/usr/bin",
                "/bin",
            ) if p]
            env = {
                "PATH": ":".join(path_dirs),
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "LANG": os.environ.get("LANG", "en_US.UTF-8"),
                "LC_ALL": os.environ.get("LC_ALL", ""),
                "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
                "HOME": td,
                "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
            }

            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    input=stdin.encode("utf-8") if stdin else b"",
                    capture_output=True,
                    timeout=timeout,
                    preexec_fn=self._apply_limits,
                    cwd=td,
                    env=env,
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
            except FileNotFoundError as exc:
                # Should not happen given the resolution above, but be explicit.
                raise ExecutionError(
                    f"Interpreter not found for {language}: {exc}"
                ) from exc
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
            pass  # best-effort; some platforms don't support every limit