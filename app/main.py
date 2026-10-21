"""FastAPI Application Entry Point."""
import os
# MUST BE AT THE VERY TOP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
# Force PyTorch to use 1 thread to prevent OpenMP crashes on Mac
torch.set_num_threads(1)

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router
from app.core.config import get_settings

logging.basicConfig(level=logging.INFO)
settings = get_settings()

app = FastAPI(
    title=settings.project_name,
    openapi_url=f"{settings.api_v1_str}/openapi.json",
    description="Unified AI Backend: Code RAG + Voice Data Analyst"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_v1_str)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the AI Backend API",
        "documentation": "Go to /docs to see the interactive Swagger UI.",
        "features": ["Code RAG", "Code Execution", "Voice Transcription", "Data Analysis"]
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    # MUST BE reload=False to prevent child process OpenMP conflicts
    uvicorn.run("app.main:app", host="127.0.0.1", port=8080, reload=False)