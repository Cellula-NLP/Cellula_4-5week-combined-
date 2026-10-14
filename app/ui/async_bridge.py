"""Bridge async code (orchestrator, httpx) to Streamlit's sync runtime.

Runs ONE background event loop for the whole process so that the cached
`httpx.AsyncClient` doesn't end up bound to a closed loop.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from typing import Any, AsyncIterator, Iterator

logger = logging.getLogger(__name__)

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    """Lazily start (once) a background event loop and return it."""
    global _loop, _thread
    with _lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            _thread = threading.Thread(
                target=_loop.run_forever,
                name="async-bridge",
                daemon=True,
            )
            _thread.start()
            logger.info("Background event loop started.")
        return _loop


def run(coro) -> Any:
    """Run a coroutine on the background loop and block for the result."""
    loop = _get_loop()
    future: Future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def iter_async(agen: AsyncIterator) -> Iterator[Any]:
    """Synchronously iterate an async generator via the background loop."""
    loop = _get_loop()
    iterator = agen.__aiter__()
    while True:
        try:
            future = asyncio.run_coroutine_threadsafe(iterator.__anext__(), loop)
            yield future.result()
        except StopAsyncIteration:
            return


def shutdown() -> None:
    """Stop the background loop (optional; called at process exit)."""
    global _loop
    with _lock:
        if _loop and not _loop.is_closed():
            _loop.call_soon_threadsafe(_loop.stop)
            _loop = None


import atexit
atexit.register(shutdown)