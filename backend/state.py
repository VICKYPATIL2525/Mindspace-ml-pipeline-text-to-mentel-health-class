"""
MindSpace Agent - Conversation State Definitions
Defines the typed state schema used by LangGraph.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


# ── Conversation Phases ──────────────────────────────────────────────────────
class Phase(str, Enum):
    START = "START"
    GREETING = "GREETING"
    CONSENT = "CONSENT"
    EMOTIONAL_CHECK = "EMOTIONAL_CHECK"
    GUIDED_CONVERSATION = "GUIDED_CONVERSATION"
    END = "END"


# ── Session Memory Schema ────────────────────────────────────────────────────
class SessionMemory(BaseModel):
    """Structured data extracted during the screening conversation."""
    consent: bool = False
    mood: str = ""
    sleep_quality: str = ""
    stress_source: str = ""
    recent_events: str = ""
    support_system: str = ""


# ── LangGraph State ──────────────────────────────────────────────────────────
class ConversationState(BaseModel):
    """Top-level state flowing through the MindSpace LangGraph."""

    # Chat messages accumulate via the add_messages reducer
    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)

    # Current phase of the screening
    phase: Phase = Phase.START

    # Structured session memory
    memory: SessionMemory = Field(default_factory=SessionMemory)

    # Latest user transcript (from STT)
    user_text: str = ""

    # Counter for guided-conversation sub-questions
    guided_step: int = 0

    class Config:
        arbitrary_types_allowed = True
