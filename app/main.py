"""Streamlit entrypoint — run with: streamlit run app/main.py"""
import logging
import os
import sys

# Ensure sandbox subprocess doesn't try to fork HF tokenizers parallelism.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _configure_logging() -> None:
    """Force INFO-level logging to reach the terminal.

    Streamlit installs its own root handler at startup, so a bare
    ``logging.basicConfig()`` is a no-op. We explicitly set the root
    level and add a StreamHandler if one isn't already attached.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Attach a stdout handler if the root logger has none.
    has_stream = any(
        isinstance(h, logging.StreamHandler)
        for h in root.handlers
    )
    if not has_stream:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        root.addHandler(handler)

    # Quiet down noisy third-party loggers we don't care about.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_logging()

from app.ui.app import main  # noqa: E402

if __name__ == "__main__":
    main()