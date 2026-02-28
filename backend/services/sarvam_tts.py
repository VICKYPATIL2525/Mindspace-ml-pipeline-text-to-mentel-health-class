"""
MindSpace Agent - Sarvam AI Text-to-Speech Service
Converts text to spoken audio using the Sarvam TTS API.
Returns base64-encoded WAV audio.
"""

from __future__ import annotations

import logging

import requests

from backend.config import SARVAM_API_KEY, SARVAM_ENDPOINT

logger = logging.getLogger(__name__)

_TTS_PATH = "/text-to-speech"


def synthesize_speech(
    text: str,
    target_language_code: str = "en-IN",
    speaker: str = "Priya",
    model: str = "bulbul:v3",
    pace: float = 1.0,
) -> str:
    """
    Convert text to speech using Sarvam AI TTS.

    Args:
        text: The text to convert to speech (max 2500 chars for bulbul:v3).
        target_language_code: BCP-47 language code for output audio.
        speaker: Voice name (e.g., Priya, Shubh, Aditya, Ritu, etc.).
        model: TTS model version.
        pace: Speech speed (0.5–2.0 for bulbul:v3).

    Returns:
        Base64-encoded WAV audio string.

    Raises:
        RuntimeError: If the API call fails.
    """
    url = f"{SARVAM_ENDPOINT}{_TTS_PATH}"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }

    # Truncate text to model limit
    if len(text) > 2500:
        text = text[:2497] + "..."

    payload = {
        "text": text,
        "target_language_code": target_language_code,
        "speaker": speaker,
        "model": model,
        "pace": pace,
        "speech_sample_rate": 24000,
    }

    try:
        logger.info("Sarvam TTS request: lang=%s speaker=%s chars=%d", target_language_code, speaker, len(text))
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        audios = data.get("audios", [])
        if not audios:
            logger.warning("Sarvam TTS returned no audio")
            return ""
        logger.info("Sarvam TTS success: audio length=%d chars", len(audios[0]))
        return audios[0]
    except requests.RequestException as exc:
        logger.error("Sarvam TTS request failed: %s", exc)
        raise RuntimeError(f"Text-to-speech failed: {exc}") from exc
