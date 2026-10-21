# app/services/audio_service.py
from __future__ import annotations
import logging

from app.core.exceptions import STTError
from app.domain.ports import SpeechToTextProvider

logger = logging.getLogger(__name__)


class AudioService:
    def __init__(self, stt_provider: SpeechToTextProvider) -> None:
        self._stt = stt_provider

    def process_audio(self, audio_path: str) -> tuple[str, str]:
        """Transcribes audio and validates the result."""
        try:
            transcript, language = self._stt.transcribe(audio_path)

            if not transcript:
                raise STTError("Could not detect any speech in the audio.")

            return transcript, language
        except Exception as exc:
            logger.error("Audio processing failed: %s", exc)
            raise STTError(str(exc)) from exc