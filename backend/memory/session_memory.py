"""
MindSpace Agent - Session Memory Manager
Provides an in-memory session store keyed by session ID.
Each session holds a ConversationState instance.
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict

from backend.state import ConversationState, SessionMemory

logger = logging.getLogger(__name__)

# ── In-memory session store ──────────────────────────────────────────────────
_sessions: Dict[str, ConversationState] = {}


def create_session() -> str:
    """Create a new session and return its unique ID."""
    session_id = uuid.uuid4().hex[:12]
    _sessions[session_id] = ConversationState()
    logger.info("Created session %s", session_id)
    return session_id


def get_session(session_id: str) -> ConversationState | None:
    """Retrieve a session by ID, or None if it doesn't exist."""
    return _sessions.get(session_id)


def update_session(session_id: str, state: ConversationState) -> None:
    """Overwrite the session state for a given session ID."""
    _sessions[session_id] = state
    logger.debug("Updated session %s — phase=%s", session_id, state.phase)


def update_memory_fields(session_id: str, fields: dict) -> None:
    """
    Merge extracted fields into the session's memory,
    only updating non-empty values.
    """
    session = get_session(session_id)
    if session is None:
        logger.warning("update_memory_fields: session %s not found", session_id)
        return

    current = session.memory.model_dump()
    for key, value in fields.items():
        if key in current and value not in (None, ""):
            current[key] = value

    session.memory = SessionMemory(**current)
    update_session(session_id, session)
    logger.info("Memory updated for session %s: %s", session_id, current)


def delete_session(session_id: str) -> None:
    """Remove a session from the store."""
    _sessions.pop(session_id, None)
    logger.info("Deleted session %s", session_id)


def list_sessions() -> list[str]:
    """Return all active session IDs (for debugging)."""
    return list(_sessions.keys())
