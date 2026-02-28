"""
MindSpace Agent - Sarvam AI Speech-to-Text Service
Sends recorded audio to the Sarvam STT API and returns a transcript.
"""

from __future__ import annotations

import logging
from io import BytesIO

import requests

from backend.config import SARVAM_API_KEY, SARVAM_ENDPOINT

logger = logging.getLogger(__name__)

# Sarvam REST endpoints
_STT_PATH = "/speech-to-text"


def transcribe_audio(
    audio_bytes: bytes,
    language_code: str = "en-IN",
    filename: str = "recording.webm",
) -> str:
    """
    Send raw audio bytes to the Sarvam AI STT API.

    Uses the saaras:v3 model with 'transcribe' mode for best results.

    Args:
        audio_bytes: Raw audio data (WAV, WebM, MP3, etc.).
        language_code: BCP-47 language code (default: en-IN).
        filename: Filename hint sent to the API.

    Returns:
        Transcribed text string.

    Raises:
        RuntimeError: If the API call fails.
    """
    url = f"{SARVAM_ENDPOINT}{_STT_PATH}"
    headers = {"api-subscription-key": SARVAM_API_KEY}

    # Determine MIME type from filename
    mime = "audio/webm"
    if filename.endswith(".wav"):
        mime = "audio/wav"
    elif filename.endswith(".mp3"):
        mime = "audio/mpeg"

    files = {"file": (filename, BytesIO(audio_bytes), mime)}
    data = {
        "model": "saaras:v3",
        "language_code": language_code,
        "mode": "transcribe",
    }

    try:
        logger.info("Sarvam STT request: url=%s lang=%s bytes=%d", url, language_code, len(audio_bytes))
        response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
        logger.info("Sarvam STT response: status=%d body=%s", response.status_code, response.text[:500])
        response.raise_for_status()
        payload = response.json()
        transcript: str = payload.get("transcript", "")
        logger.info("Sarvam STT transcript: %s", transcript)
        return transcript.strip()
    except requests.RequestException as exc:
        logger.error("Sarvam STT request failed: %s", exc)
        raise RuntimeError(f"Speech-to-text failed: {exc}") from exc
