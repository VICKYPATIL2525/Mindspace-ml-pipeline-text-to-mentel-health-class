"""
MindSpace Agent - LangGraph Conversation Graph
Defines the state-machine that orchestrates the screening conversation.

States:
    START → GREETING → CONSENT → EMOTIONAL_CHECK → GUIDED_CONVERSATION → END
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END as GRAPH_END

from backend.state import ConversationState, Phase, SessionMemory
from backend.services.llm_service import (
    chat_completion,
    extract_session_fields,
    load_prompt,
    load_skill,
)

logger = logging.getLogger(__name__)

# Total number of guided sub-questions before we finish
_GUIDED_TOTAL_STEPS = 5


# ══════════════════════════════════════════════════════════════════════════════
# Node functions — each receives & returns ConversationState
# ══════════════════════════════════════════════════════════════════════════════

def greeting_node(state: ConversationState) -> Dict[str, Any]:
    """Generate the greeting message and advance phase."""
    system = load_prompt("system_prompt")
    skill = load_skill("greeting")
    prompt = f"{system}\n\n---\n\n{skill}"

    reply = chat_completion(
        system_prompt=prompt,
        messages=[],
        temperature=0.7,
    )

    return {
        "messages": [AIMessage(content=reply)],
        "phase": Phase.CONSENT,  # next expected phase
    }


def consent_node(state: ConversationState) -> Dict[str, Any]:
    """Ask for consent and explain the AI disclaimer."""
    system = load_prompt("system_prompt")
    skill = load_skill("consent")
    prompt = f"{system}\n\n---\n\n{skill}"

    history = _messages_to_dicts(state.messages)

    reply = chat_completion(
        system_prompt=prompt,
        messages=history,
        temperature=0.7,
    )

    return {
        "messages": [AIMessage(content=reply)],
        # Phase stays CONSENT until user responds; router decides next
    }


def process_consent_node(state: ConversationState) -> Dict[str, Any]:
    """Evaluate the user's consent answer and update memory."""
    history = _messages_to_dicts(state.messages)
    extracted = extract_session_fields(history, state.memory.model_dump())

    consent_value = extracted.get("consent")
    new_memory = state.memory.model_copy()

    if consent_value is True:
        new_memory.consent = True
        return {"memory": new_memory, "phase": Phase.EMOTIONAL_CHECK}
    elif consent_value is False:
        new_memory.consent = False
        return {"memory": new_memory, "phase": Phase.END}
    else:
        # Ambiguous — stay in consent to re-ask
        return {"memory": new_memory, "phase": Phase.CONSENT}


def emotional_check_node(state: ConversationState) -> Dict[str, Any]:
    """Ask the emotional baseline question."""
    system = load_prompt("system_prompt")
    skill = load_skill("emotion_check")
    empathy = load_skill("empathy")
    prompt = f"{system}\n\n---\n\n{skill}\n\n---\n\n{empathy}"

    history = _messages_to_dicts(state.messages)

    reply = chat_completion(
        system_prompt=prompt,
        messages=history,
        temperature=0.7,
    )

    return {"messages": [AIMessage(content=reply)]}


def process_emotion_node(state: ConversationState) -> Dict[str, Any]:
    """Extract emotional signals from the user's response."""
    history = _messages_to_dicts(state.messages)
    extracted = extract_session_fields(history, state.memory.model_dump())

    new_memory = _merge_memory(state.memory, extracted)
    return {"memory": new_memory, "phase": Phase.GUIDED_CONVERSATION, "guided_step": 0}


def guided_conversation_node(state: ConversationState) -> Dict[str, Any]:
    """Ask the next guided sub-question."""
    system = load_prompt("system_prompt")
    skill = load_skill("guided_conversation")
    empathy = load_skill("empathy")
    prompt = (
        f"{system}\n\n---\n\n{skill}\n\n---\n\n{empathy}\n\n"
        f"You are currently on guided question {state.guided_step + 1} of {_GUIDED_TOTAL_STEPS}. "
        f"Ask ONLY the question for this step. Do not repeat earlier questions."
    )

    history = _messages_to_dicts(state.messages)

    reply = chat_completion(
        system_prompt=prompt,
        messages=history,
        temperature=0.7,
    )

    return {"messages": [AIMessage(content=reply)]}


def process_guided_node(state: ConversationState) -> Dict[str, Any]:
    """Extract info from guided response and advance the step counter."""
    history = _messages_to_dicts(state.messages)
    extracted = extract_session_fields(history, state.memory.model_dump())

    new_memory = _merge_memory(state.memory, extracted)
    next_step = state.guided_step + 1

    if next_step >= _GUIDED_TOTAL_STEPS:
        return {"memory": new_memory, "phase": Phase.END, "guided_step": next_step}
    else:
        return {"memory": new_memory, "guided_step": next_step}


def end_node(state: ConversationState) -> Dict[str, Any]:
    """Generate a closing message."""
    system = load_prompt("system_prompt")

    if not state.memory.consent:
        closing = (
            "Thank you for your time. If you ever feel ready to talk, "
            "I'll be here. Take care of yourself. 💛"
        )
    else:
        closing_prompt = (
            f"{system}\n\nThe screening is now complete. "
            "Thank the user sincerely for sharing. Summarise what you heard "
            "briefly and encourage them to seek professional support if needed. "
            "Remind them this was not a diagnosis."
        )
        history = _messages_to_dicts(state.messages)
        closing = chat_completion(
            system_prompt=closing_prompt,
            messages=history,
            temperature=0.7,
        )

    return {"messages": [AIMessage(content=closing)], "phase": Phase.END}


# ══════════════════════════════════════════════════════════════════════════════
# Router functions
# ══════════════════════════════════════════════════════════════════════════════

def route_after_consent_process(state: ConversationState) -> str:
    """Decide the next node after processing consent."""
    if state.phase == Phase.EMOTIONAL_CHECK:
        return "emotional_check"
    elif state.phase == Phase.END:
        return "end"
    else:
        return "consent"  # re-ask


def route_after_guided_process(state: ConversationState) -> str:
    """Decide whether to continue guided conversation or end."""
    if state.phase == Phase.END:
        return "end"
    return "guided_conversation"


# ══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """
    Construct and compile the MindSpace conversation graph.

    The graph alternates between AI-speaking nodes (which generate a response
    and then **pause** to wait for user input) and processing nodes (which
    analyse the user's reply and route to the next step).
    """
    graph = StateGraph(ConversationState)

    # ── Add nodes ────────────────────────────────────────────────────────
    graph.add_node("greeting", greeting_node)
    graph.add_node("consent", consent_node)
    graph.add_node("process_consent", process_consent_node)
    graph.add_node("emotional_check", emotional_check_node)
    graph.add_node("process_emotion", process_emotion_node)
    graph.add_node("guided_conversation", guided_conversation_node)
    graph.add_node("process_guided", process_guided_node)
    graph.add_node("end", end_node)

    # ── Set entry point ──────────────────────────────────────────────────
    graph.set_entry_point("greeting")

    # ── Edges ────────────────────────────────────────────────────────────
    # greeting → consent (AI speaks, then we wait for user input;
    #   the next call from the API will invoke process_consent after adding HumanMessage)
    graph.add_edge("greeting", "consent")

    # consent → (pause for user input) — handled by the agent loop
    # When user replies, we invoke process_consent
    graph.add_edge("consent", GRAPH_END)           # pause point
    graph.add_edge("process_consent", GRAPH_END)    # pause point after routing is manual
    graph.add_edge("emotional_check", GRAPH_END)    # pause point
    graph.add_edge("process_emotion", GRAPH_END)    # pause point
    graph.add_edge("guided_conversation", GRAPH_END)  # pause point
    graph.add_edge("process_guided", GRAPH_END)     # pause point
    graph.add_edge("end", GRAPH_END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _messages_to_dicts(messages: list) -> list[dict[str, str]]:
    """Convert LangChain message objects to plain dicts for the LLM service."""
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result


def _merge_memory(current: SessionMemory, extracted: dict) -> SessionMemory:
    """Merge extracted fields into existing memory, keeping non-empty values."""
    data = current.model_dump()
    for key, value in extracted.items():
        if key in data and value not in (None, ""):
            data[key] = value
    return SessionMemory(**data)
