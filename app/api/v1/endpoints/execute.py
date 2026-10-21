from fastapi import APIRouter, Depends, HTTPException
from app.api.dependencies import get_code_executor
from app.domain.models import CodeExecuteRequest
from app.services.code_executor import CodeExecutor

router = APIRouter()

@router.post("/run")
async def execute_code(
    request: CodeExecuteRequest,
    executor: CodeExecutor = Depends(get_code_executor)
):
    try:
        result = executor.execute(request.code, request.language)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))