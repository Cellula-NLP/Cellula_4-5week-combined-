"""Wipe the Chroma collection and SQLite session DB.

Usage:
    python scripts/reset_db.py [--yes]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Skip confirmation.")
    args = parser.parse_args()

    settings = get_settings()

    if not args.yes:
        confirm = input(
            f"This will DELETE:\n  • {settings.chroma_path}\n  • {settings.sessions_db}\n"
            "Continue? [y/N] "
        ).strip().lower()
        if confirm != "y":
            print("Aborted.")
            return 1

    for path in (settings.chroma_path, settings.sessions_db):
        p = Path(path)
        if p.is_dir():
            shutil.rmtree(p)
            print(f"Removed directory: {p}")
        elif p.is_file():
            p.unlink()
            print(f"Removed file: {p}")
        else:
            print(f"Skipped (not found): {p}")

    print("✅ Reset complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())