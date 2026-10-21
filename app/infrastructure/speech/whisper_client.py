"""Faster-Whisper implementation of the SpeechToTextProvider."""
from __future__ import annotations

import logging
import os
from faster_whisper import WhisperModel

from app.core.config import get_settings
from app.core.exceptions import STTError
from app.domain.ports import SpeechToTextProvider

logger = logging.getLogger(__name__)


class FasterWhisperClient(SpeechToTextProvider):
    """Wraps the Faster-Whisper model for Arabic/English transcription."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model_size = settings.whisper_model_size
        self._device = settings.whisper_device
        self._compute_type = settings.whisper_compute_type
        self._model: WhisperModel | None = None

    def _load_model(self) -> WhisperModel:
        """Lazy load the model only when the first audio file arrives."""
        if self._model is not None:
            return self._model

        logger.info(
            "Loading Faster-Whisper model '%s' (device=%s, compute=%s)…",
            self._model_size, self._device, self._compute_type
        )
        try:
            # This downloads the model on first run and caches it locally.
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type
            )
            logger.info("Whisper model loaded successfully.")
            return self._model
        except Exception as exc:
            logger.error("Failed to load Whisper model: %s", exc)
            raise STTError(f"Could not load Whisper model: {exc}") from exc

    def transcribe(self, audio_path: str) -> tuple[str, str]:
        """
        Transcribes the audio file.
        Returns a tuple of (transcript_text, detected_language_code).
        """
        if not os.path.exists(audio_path):
            raise STTError(f"Audio file not found: {audio_path}")

        model = self._load_model()

        try:
            # beam_size=5 provides a good balance between speed and accuracy.
            # vad_filter=True filters out silence so we don't transcribe background noise.
            segments, info = model.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True
            )

            # segments is a generator. We must join the text from each segment.
            transcript = " ".join([segment.text for segment in segments]).strip()
            detected_language = info.language

            if not transcript:
                logger.warning("Whisper returned an empty transcript. (Silence or unrecognized audio).")
                return "", detected_language

            logger.info(
                "Transcribed audio (lang=%s, prob=%.2f): %s",
                detected_language, info.language_probability, transcript[:100]
            )
            return transcript, detected_language

        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            raise STTError(f"Transcription failed: {exc}") from exc