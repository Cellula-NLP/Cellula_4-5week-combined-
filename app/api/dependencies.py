from functools import lru_cache
from app.infrastructure.llm.openrouter_client import OpenRouterClient
from app.infrastructure.vectorstore.chroma_store import ChromaVectorStore
from app.infrastructure.speech.whisper_client import FasterWhisperClient
from app.infrastructure.database.sqlite_manager import SQLiteManager
from app.infrastructure.tools.code_runner import SubprocessSandbox
from app.infrastructure.web_search.tavily_client import TavilySearchClient
from app.infrastructure.embeddings.sentence_transformer import SentenceTransformerEmbedder
from app.infrastructure.tools.reranker import CrossEncoderReranker
from app.infrastructure.loaders.chunker import CodeAwareChunker
from app.infrastructure.memory.session_memory import SQLiteMemoryStore
from app.core.config import get_settings

from app.services.audio_service import AudioService
from app.services.dataset_service import DatasetService
from app.services.sql_service import SQLService
from app.services.generator import CodeGenerator
from app.services.explainer import CodeExplainer
from app.services.code_executor import CodeExecutor
from app.services.retriever import Retriever
from app.services.relevance_evaluator import RelevanceEvaluator
from app.services.ingestion import IngestionService
from app.services.web_search_service import WebSearchService

# ── Core Infrastructure ──
@lru_cache()
def get_llm() -> OpenRouterClient:
    return OpenRouterClient()

@lru_cache()
def get_vdb() -> ChromaVectorStore:
    s = get_settings()
    return ChromaVectorStore(s.chroma_path, s.chroma_collection)

@lru_cache()
def get_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()

@lru_cache()
def get_stt() -> FasterWhisperClient:
    return FasterWhisperClient()

@lru_cache()
def get_db() -> SQLiteManager:
    return SQLiteManager()

@lru_cache()
def get_memory_store() -> SQLiteMemoryStore:
    s = get_settings()
    return SQLiteMemoryStore(s.sessions_db)

@lru_cache()
def get_web_search_client() -> TavilySearchClient:
    return TavilySearchClient(get_settings().tavily_api_key)

@lru_cache()
def get_chunker() -> CodeAwareChunker:
    return CodeAwareChunker()

# ── RAG Singletons ──
@lru_cache()
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()

@lru_cache()
def get_retriever() -> Retriever:
    return Retriever(
        embedder=get_embedder(),
        vectorstore=get_vdb(),
        reranker=get_reranker()
    )

@lru_cache()
def get_evaluator() -> RelevanceEvaluator:
    return RelevanceEvaluator(llm=get_llm())

@lru_cache()
def get_ingestion() -> IngestionService:
    return IngestionService(
        embedder=get_embedder(),
        vectorstore=get_vdb(),
        chunker=get_chunker()
    )

@lru_cache()
def get_web_search_service() -> WebSearchService:
    return WebSearchService(
        search_client=get_web_search_client(),
        llm=get_llm(),
        ingestion=get_ingestion()
    )

# ── Services ──
def get_audio_service() -> AudioService:
    return AudioService(stt_provider=get_stt())

def get_dataset_service() -> DatasetService:
    return DatasetService(db=get_db())

def get_sql_service() -> SQLService:
    return SQLService(llm=get_llm(), db=get_db())

def get_code_generator() -> CodeGenerator:
    return CodeGenerator(llm=get_llm(), memory=get_memory_store())

def get_code_explainer() -> CodeExplainer:
    return CodeExplainer(llm=get_llm(), memory=get_memory_store())

def get_code_executor() -> CodeExecutor:
    return CodeExecutor(SubprocessSandbox())