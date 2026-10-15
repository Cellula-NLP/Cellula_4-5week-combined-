"""Streamlit entrypoint — run with: streamlit run app/main.py"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from app.ui.app import main

if __name__ == "__main__":
    main()