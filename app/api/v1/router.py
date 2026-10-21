from fastapi import APIRouter
from app.api.v1.endpoints import code, execute, audio, dataset, query

api_router = APIRouter()
api_router.include_router(code.router, prefix="/code", tags=["Code RAG"])
api_router.include_router(execute.router, prefix="/execute", tags=["Code Execution"])
api_router.include_router(audio.router, prefix="/audio", tags=["Voice"])
api_router.include_router(dataset.router, prefix="/dataset", tags=["Dataset"])
api_router.include_router(query.router, prefix="/query", tags=["SQL Query"])