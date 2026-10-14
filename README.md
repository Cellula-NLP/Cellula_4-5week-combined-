# AI Coding Assistant

Intelligent AI coding assistant with:
- LLM-based intent classification (Explain vs Generate)
- RAG over HumanEval dataset using Chroma
- Relevance evaluation before generation
- Sandboxed code execution
- Human-feedback learning loop
- Streamlit UI with streaming, syntax highlighting, and run button

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env .env.local  # edit LLM_API_KEY
python scripts/seed_humaneval.py
streamlit run app/main.py