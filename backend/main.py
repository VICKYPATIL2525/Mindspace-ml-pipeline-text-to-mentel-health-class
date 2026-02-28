"""
MindSpace Agent - FastAPI Application Entry Point
Exposes REST endpoints for the voice-based mental health screening agent.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import validate_config
from backend.conversation_agent import (
    start_conversation,
    handle_user_message,
    get_session_summary,
)
from backend.services.sarvam_stt import transcribe_audio
from backend.services.sarvam_tts import synthesize_speech

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Validate configuration on startup ────────────────────────────────────────
validate_config()

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="MindSpace Voice Agent",
    description="Voice-based AI mental health screening agent",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────────────────
class StartResponse(BaseModel):
    session_id: str
    message: str
    audio_base64: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: str
    text: str


class ChatResponse(BaseModel):
    message: str
    is_ended: bool
    memory: Optional[dict] = None
    transcript: Optional[str] = None
    audio_base64: Optional[str] = None


class SummaryResponse(BaseModel):
    session_id: str
    memory: dict


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def root():
    return {"status": "ok", "agent": "MindSpace Voice Agent v0.1.0"}


@app.post("/api/start", response_model=StartResponse)
async def api_start(language_code: str = "en-IN"):
    """Start a new screening session. Returns session ID, greeting, and audio."""
    try:
        session_id, greeting = start_conversation()
        # Generate TTS for the greeting
        audio_b64 = _safe_tts(greeting, language_code)
        return StartResponse(session_id=session_id, message=greeting, audio_base64=audio_b64)
    except Exception as exc:
        logger.exception("Failed to start session")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    """Send a text message to an existing session."""
    try:
        reply, is_ended = handle_user_message(req.session_id, req.text)
        memory = get_session_summary(req.session_id)
        audio_b64 = _safe_tts(reply, "en-IN")
        return ChatResponse(message=reply, is_ended=is_ended, memory=memory, audio_base64=audio_b64)
    except Exception as exc:
        logger.exception("Chat error for session %s", req.session_id)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/voice", response_model=ChatResponse)
async def api_voice(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
    language_code: str = Form("en-IN"),
):
    """
    Accept audio from the frontend, transcribe via Sarvam STT,
    then process through the conversation agent. Returns AI reply + TTS audio.
    """
    try:
        audio_bytes = await audio.read()
        transcript = transcribe_audio(
            audio_bytes=audio_bytes,
            language_code=language_code,
            filename=audio.filename or "recording.webm",
        )
        if not transcript:
            fallback_msg = "I couldn't catch that. Could you please try again?"
            audio_b64 = _safe_tts(fallback_msg, language_code)
            return ChatResponse(
                message=fallback_msg,
                is_ended=False,
                audio_base64=audio_b64,
            )

        reply, is_ended = handle_user_message(session_id, transcript)
        memory = get_session_summary(session_id)
        audio_b64 = _safe_tts(reply, language_code)
        return ChatResponse(
            message=reply,
            is_ended=is_ended,
            memory=memory,
            transcript=transcript,
            audio_base64=audio_b64,
        )
    except RuntimeError as exc:
        logger.error("STT error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.exception("Voice endpoint error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/summary/{session_id}", response_model=SummaryResponse)
async def api_summary(session_id: str):
    """Retrieve the structured session memory."""
    memory = get_session_summary(session_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SummaryResponse(session_id=session_id, memory=memory)


# ── TTS helper ───────────────────────────────────────────────────────────────
def _safe_tts(text: str, language_code: str = "en-IN") -> str | None:
    """Generate TTS audio, returning None on failure instead of crashing."""
    try:
        return synthesize_speech(text=text, target_language_code=language_code)
    except Exception as exc:
        logger.warning("TTS failed (non-fatal): %s", exc)
        return None


# ── Mount static frontend ────────────────────────────────────────────────────
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")
