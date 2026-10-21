import os, tempfile
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from starlette.concurrency import run_in_threadpool
from app.api.dependencies import get_audio_service
from app.domain.models import TranscriptionResponse
from app.services.audio_service import AudioService

router = APIRouter()

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    audio_service: AudioService = Depends(get_audio_service),
):
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.ogg')):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        transcript, language = await run_in_threadpool(audio_service.process_audio, tmp_path)
        return TranscriptionResponse(transcript=transcript, detected_language=language)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)