#  AI Backend

An intelligent, production-grade AI backend built with **FastAPI**, serving two distinct capabilities via a single API:
1. **AI Coding Assistant (RAG):** Explains code, generates code using a Chroma Vector DB, autonomously searches Stack Overflow for missing knowledge (via Tavily), and securely executes code in a sandbox.
2. **Voice Data Analyst:** Transcribes Arabic/English audio to text (Faster-Whisper), translates natural language into SQL queries, and securely analyzes uploaded CSV datasets via SQLite.

## ✨ Core Features

- **🧠 Intelligent Routing:** LLM-based classification (Explain vs. Generate) with heuristic fallbacks to save latency and API costs.
- **🔍 Advanced RAG Pipeline:** 3-stage retrieval (Bi-encoder → Threshold → Cross-encoder Reranking) for highly accurate code context.
- **🌐 Autonomous Learning Loop:** If local ChromaDB lacks the answer, the system automatically searches Stack Overflow (Tavily), extracts valid code, and ingests it for future queries.
- **🎙️ Multilingual Voice-to-SQL:** Uses Faster-Whisper (int8 CPU optimized) to detect Arabic or English, generating valid SQLite queries.
- **🛡️ Zero-Trust Security:** 
  - SQL queries are validated using `sqlglot` (AST parsing) to strictly prevent `DROP/DELETE` commands.
  - Code execution runs in a Subprocess Sandbox with strict `RLIMIT` constraints (CPU, RAM, Process limits).
- **💾 Stateful Memory:** SQLite-backed conversation memory persists across API calls using `session_id`.
- **⚡ Fully Async & Swagger Documented:** Built on FastAPI with automatic OpenAPI docs at `/docs`.

## 🛠️ Tech Stack

- **Backend:** FastAPI, Uvicorn, Pydantic
- **LLMs:** Groq / OpenRouter (Llama-3.3-70b)
- **RAG & VDB:** ChromaDB, Sentence-Transformers (Jina Embeddings, Cross-Encoders)
- **Audio:** Faster-Whisper (CTranslate2)
- **Database:** SQLite3, Pandas, SQLGlot
- **Web Search:** Tavily API

## 🚀 Quick Start

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e
put your LLM_API_KEY
python scripts/seed_humaneval.py
streamlit run app/main.py
