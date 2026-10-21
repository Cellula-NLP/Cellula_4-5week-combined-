import os, tempfile
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from starlette.concurrency import run_in_threadpool
from app.api.dependencies import get_dataset_service
from app.domain.models import DatasetUploadResponse
from app.services.dataset_service import DatasetService

router = APIRouter()

@router.post("/upload", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        table_name, row_count, schema = await run_in_threadpool(
            dataset_service.process_upload, tmp_path, file.filename
        )
        return DatasetUploadResponse(
            filename=file.filename, table_name=table_name, row_count=row_count, columns=schema
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)