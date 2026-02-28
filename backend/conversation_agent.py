"""
MindSpace Agent - Conversation Agent
High-level orchestrator that ties together the LangGraph, session memory,
and the request/response cycle from the API layer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from langchain_core.messages import AIMessage, HumanMessage

from backend.state import ConversationState, Phase
from backend.graph.mindspace_graph import (
    build_graph,
    process_consent_node,
    process_emotion_node,
    process_guided_node,
    route_after_consent_process,
    route_after_guided_process,
    emotional_check_node,
    guided_conversation_node,
    end_node,
)
from backend.memory.session_memory import (
    create_session,
    get_session,
    update_session,
)

logger = logging.getLogger(__name__)

# Build the compiled graph once at module level
_graph = build_graph()


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def start_conversation() -> Tuple[str, str]:
    """
    Begin a new screening session.

    Returns:
        (session_id, ai_greeting_text)
    """
    session_id = create_session()
    state = get_session(session_id)

    # Run the graph from entry point (greeting → consent)
    result = _graph.invoke(state.model_dump())
    new_state = ConversationState(**result)

    update_session(session_id, new_state)

    # The last AI message is the greeting + consent prompt
    ai_text = _last_ai_text(new_state)
    logger.info("Session %s started — greeting sent", session_id)
    return session_id, ai_text


def handle_user_message(session_id: str, user_text: str) -> Tuple[str, bool]:
    """
    Process a user message within an existing session.

    Args:
        session_id: Active session identifier.
        user_text: Transcribed user speech.

    Returns:
        (ai_response_text, is_ended)
    """
    state = get_session(session_id)
    if state is None:
        return "Session not found. Please start a new screening.", True

    # Append the user's message
    state.messages.append(HumanMessage(content=user_text))
    state.user_text = user_text

    # Determine which processing step to run based on current phase
    phase = state.phase

    if phase == Phase.CONSENT:
        state = _run_node_fn(process_consent_node, state)
        # Route after consent processing
        route = route_after_consent_process(state)
        if route == "emotional_check":
            state = _run_node_fn(emotional_check_node, state)
        elif route == "end":
            state = _run_node_fn(end_node, state)
        else:
            # Re-ask consent
            from backend.graph.mindspace_graph import consent_node
            state = _run_node_fn(consent_node, state)

    elif phase == Phase.EMOTIONAL_CHECK:
        state = _run_node_fn(process_emotion_node, state)
        # After emotion processing, go to guided conversation
        state = _run_node_fn(guided_conversation_node, state)

    elif phase == Phase.GUIDED_CONVERSATION:
        state = _run_node_fn(process_guided_node, state)
        route = route_after_guided_process(state)
        if route == "end":
            state = _run_node_fn(end_node, state)
        else:
            state = _run_node_fn(guided_conversation_node, state)

    elif phase == Phase.END:
        # Already ended
        return "The screening session has ended. Thank you for your time. 💛", True

    update_session(session_id, state)

    ai_text = _last_ai_text(state)
    is_ended = state.phase == Phase.END
    logger.info(
        "Session %s — phase=%s, ended=%s", session_id, state.phase, is_ended
    )
    return ai_text, is_ended


def get_session_summary(session_id: str) -> Dict[str, Any] | None:
    """Return the session memory as a dict, or None if session doesn't exist."""
    state = get_session(session_id)
    if state is None:
        return None
    return state.memory.model_dump()


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _run_node_fn(node_fn, state: ConversationState) -> ConversationState:
    """Run a single graph node function and merge the returned updates."""
    updates = node_fn(state)
    data = state.model_dump()

    for key, value in updates.items():
        if key == "messages":
            # Append new messages
            data["messages"] = data.get("messages", []) + [
                m if isinstance(m, dict) else _msg_to_dict(m) for m in value
            ]
        else:
            data[key] = value if not hasattr(value, "value") else value.value

    return ConversationState(**data)


def _msg_to_dict(msg) -> dict:
    """Convert a LangChain message to a serialisable dict."""
    if isinstance(msg, AIMessage):
        return {"role": "assistant", "content": msg.content, "type": "ai"}
    elif isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content, "type": "human"}
    return {"role": "unknown", "content": str(msg)}


def _last_ai_text(state: ConversationState) -> str:
    """Extract the text of the last AI message in the state."""
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage):
            return msg.content
        if isinstance(msg, dict) and msg.get("type") == "ai":
            return msg.get("content", "")
    return ""
