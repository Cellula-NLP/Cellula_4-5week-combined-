from fastapi import APIRouter, Depends, HTTPException
from app.api.dependencies import get_sql_service
from app.domain.models import AnalyzeRequest, AnalyzeResponse
from app.services.sql_service import SQLService

router = APIRouter()

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(
    request: AnalyzeRequest,
    sql_service: SQLService = Depends(get_sql_service),
):
    try:
        result = await sql_service.analyze(request.user_text, request.dataset_name)
        return AnalyzeResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))