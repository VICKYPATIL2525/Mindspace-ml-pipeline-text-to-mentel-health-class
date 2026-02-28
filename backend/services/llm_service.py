"""
MindSpace Agent - Azure OpenAI LLM Service
Provides chat-completion and structured-extraction helpers via Azure OpenAI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from openai import AzureOpenAI

from backend.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    PROMPTS_DIR,
    SKILLS_DIR,
)

logger = logging.getLogger(__name__)

# ── Azure OpenAI client (singleton) ─────────────────────────────────────────
_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )
    return _client


# ── Helpers ──────────────────────────────────────────────────────────────────
def _load_md(path: Path) -> str:
    """Read a Markdown file and return its content."""
    return path.read_text(encoding="utf-8")


def load_skill(skill_name: str) -> str:
    """Load a skill prompt from backend/skills/<name>.md."""
    return _load_md(SKILLS_DIR / f"{skill_name}.md")


def load_prompt(prompt_name: str) -> str:
    """Load a prompt from backend/prompts/<name>.md."""
    return _load_md(PROMPTS_DIR / f"{prompt_name}.md")


# ── Chat completion ──────────────────────────────────────────────────────────
def chat_completion(
    system_prompt: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """
    Call Azure OpenAI chat completion.

    Args:
        system_prompt: The system-level instruction.
        messages: List of {"role": ..., "content": ...} dicts.
        temperature: Sampling temperature.
        max_tokens: Max tokens for the response.

    Returns:
        The assistant's reply text.
    """
    client = _get_client()
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    reply = response.choices[0].message.content or ""
    logger.info("LLM reply (first 120 chars): %s", reply[:120])
    return reply.strip()


# ── Structured extraction ────────────────────────────────────────────────────
def extract_session_fields(
    conversation_history: list[dict[str, str]],
    current_memory: dict[str, Any],
) -> dict[str, Any]:
    """
    Use the LLM to extract / update structured session fields from the
    latest user message in context of the full conversation.

    Returns a dict matching the SessionMemory schema.
    """
    extraction_prompt = load_prompt("extraction_prompt")

    # Build the user message that includes current memory for context
    user_msg = (
        f"Current memory state:\n```json\n{json.dumps(current_memory, indent=2)}\n```\n\n"
        "Based on the conversation above, return the updated JSON."
    )

    messages = conversation_history + [{"role": "user", "content": user_msg}]

    raw = chat_completion(
        system_prompt=extraction_prompt,
        messages=messages,
        temperature=0.0,
        max_tokens=300,
    )

    # Parse the JSON from the LLM response
    try:
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        return json.loads(cleaned.strip())
    except (json.JSONDecodeError, IndexError):
        logger.warning("Failed to parse extraction JSON: %s", raw)
        return current_memory
