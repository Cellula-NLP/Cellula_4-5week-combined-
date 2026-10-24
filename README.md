# AI Backend

A single FastAPI backend handling two distinct AI workflows:
1. **RAG Coding Assistant:** Explains code, generates code, and executes it securely. It uses a local ChromaDB and falls back to autonomous Stack Overflow searches if knowledge is missing.
2. **Voice Data Analyst:** Transcribes Arabic/English speech, converts natural language into SQLite queries, and analyzes uploaded CSVs securely.

## Architecture & Design Choices

This project is built using a Hexagonal (Ports & Adapters) architecture to keep business logic isolated from external tools.

- **Dual-Workflow Routing:** The API handles RAG (code) and SQL (data) independently. The code generation path uses a 3-stage retrieval pipeline (Bi-encoder search → Threshold filtering → Cross-encoder reranking) for highly accurate context. 
- **Autonomous Learning Loop:** If a user asks for code that isn't in the local ChromaDB, the system automatically queries Stack Overflow via Tavily. It extracts the valid code, ingests it into ChromaDB for future use, and generates the answer.
- **Zero-Trust SQL Execution:** The LLM is not trusted to generate safe SQL. All queries are parsed via `sqlglot` (AST) to strictly enforce `SELECT` statements and block destructive commands (DROP, DELETE). 
- **Secure Code Execution:** Generated code runs in a strict subprocess sandbox with OS-level `RLIMIT` constraints (CPU, RAM, process limits) and a scrubbed environment.
- **Stateful Memory:** Because REST APIs are stateless, a SQLite-backed `session_id` is used to persist chat history and user preferences across multiple requests.
- **Mac OpenMP Fix:** PyTorch and Faster-Whisper (CTranslate2) clash on macOS. This is resolved by forcing single-threaded execution (`torch.set_num_threads(1)`) and isolating the event loop.

## Quick Start

```bash
# 1. Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure environment
# Create a .env file with your keys:
# LLM_API_KEY="gsk_your_groq_key"
# TAVILY_API_KEY="tvly-your_tavily_key"

# 3. Seed the local vector database (HumanEval dataset)
python scripts/seed_humaneval.py

# 4. Start the server
python app/main.py