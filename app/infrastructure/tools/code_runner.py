"""Sandboxed code execution via subprocess with resource limits."""
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


# ── SQL Wrapper Code ────────────────────────────────────────────
# This Python script will be generated on the fly to execute the SQL query.
_SQL_WRAPPER = """
import sqlite3
import sys

def main():
    if len(sys.argv) < 2:
        return
    filepath = sys.argv[1]
    with open(filepath, "r", encoding="utf-8") as f:
        sql_code = f.read()

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    try:
        # Execute the SQL code
        cursor.executescript(sql_code)
        conn.commit()

        # If the last statement was a SELECT, fetch and print results
        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            print("\\t".join(columns))
            print("-" * 40)
            for row in cursor.fetchall():
                print("\\t".join(str(val) for val in row))
    except sqlite3.Error as e:
        print(f"SQL Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
"""

# ── Interpreter resolution ──────────────────────────────────────
def _resolve_command(*candidates: str) -> str | None:
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


_PYTHON_BIN = _resolve_command(sys.executable or "", "python3", "python")
_NODE_BIN = _resolve_command("node", "nodejs")
_BASH_BIN = _resolve_command("bash", "/bin/bash")

_EXEC_COMMANDS: dict[str, list[str]] = {}
if _PYTHON_BIN:
    _EXEC_COMMANDS["python"] = [_PYTHON_BIN, "-u"]
if _NODE_BIN:
    _EXEC_COMMANDS["javascript"] = [_NODE_BIN]
if _BASH_BIN:
    _EXEC_COMMANDS["bash"] = [_BASH_BIN]

_VENV_BIN = str(Path(sys.executable).parent) if sys.executable else ""


class SubprocessSandbox(CodeRunner):
    def __init__(self, default_timeout_s: float = 5.0, mem_mb: int = 256) -> None:
        self._timeout = default_timeout_s
        self._mem_bytes = mem_mb * 1024 * 1024

    @property
    def available_languages(self) -> frozenset[str]:
        # Add 'sql' to the available languages if Python is installed
        langs = set(_EXEC_COMMANDS.keys())
        if _PYTHON_BIN:
            langs.add("sql")
        return frozenset(langs)

    def run(
        self,
        code: str,
        language: str,
        *,
        timeout_s: float | None = None,
        stdin: str = "",
    ) -> ExecutionResult:
        language = language.lower()

        if language == "sql" and _PYTHON_BIN:
            return self._run_sql(code, timeout_s, stdin)

        if language not in _EXEC_COMMANDS:
            available = ", ".join(sorted(self.available_languages)) or "(none detected)"
            raise ExecutionError(
                f"Unsupported language: {language}. Available: {available}"
            )

        timeout = timeout_s or self._timeout
        suffix = self._suffix_for(language)

        with tempfile.TemporaryDirectory(prefix="sandbox_") as td:
            path = Path(td) / f"main{suffix}"
            path.write_text(code, encoding="utf-8")
            cmd = _EXEC_COMMANDS[language] + [str(path)]

            return self._execute_subprocess(cmd, code, timeout, td)

    def _run_sql(self, code: str, timeout_s: float | None, stdin: str) -> ExecutionResult:
        """Runs SQL by dynamically generating a Python wrapper."""
        timeout = timeout_s or self._timeout
        with tempfile.TemporaryDirectory(prefix="sandbox_sql_") as td:
            # 1. Write the SQL code
            sql_path = Path(td) / "query.sql"
            sql_path.write_text(code, encoding="utf-8")

            # 2. Write the Python wrapper
            wrapper_path = Path(td) / "wrapper.py"
            wrapper_path.write_text(_SQL_WRAPPER, encoding="utf-8")

            # 3. Run the wrapper, passing the SQL file as an argument
            cmd = [_PYTHON_BIN, "-u", str(wrapper_path), str(sql_path)]
            return self._execute_subprocess(cmd, stdin, timeout, td)

    def _execute_subprocess(self, cmd: list[str], stdin: str, timeout: float, td: str) -> ExecutionResult:
        path_dirs = [p for p in (_VENV_BIN, "/usr/local/bin", "/usr/bin", "/bin") if p]
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
        except Exception as exc:
            raise ExecutionError(f"Sandbox failure: {exc}") from exc

    @staticmethod
    def _suffix_for(language: str) -> str:
        return {
            "python": ".py",
            "javascript": ".js",
            "bash": ".sh",
            "sql": ".sql",
        }.get(language, ".txt")

    def _apply_limits(self) -> None:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
            resource.setrlimit(resource.RLIMIT_AS, (self._mem_bytes, self._mem_bytes))
            resource.setrlimit(resource.RLIMIT_FSIZE, (5 * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            os.setsid()
        except Exception:
            pass