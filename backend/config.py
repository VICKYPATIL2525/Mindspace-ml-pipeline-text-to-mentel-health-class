"""
MindSpace Agent - Configuration Module
Loads environment variables and provides centralised config access.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# ── Azure OpenAI ─────────────────────────────────────────────────────────────
AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

# ── Sarvam AI ────────────────────────────────────────────────────────────────
SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
SARVAM_ENDPOINT: str = os.getenv("SARVAM_ENDPOINT", "https://api.sarvam.ai/v1")

# ── Paths ────────────────────────────────────────────────────────────────────
SKILLS_DIR: Path = ROOT_DIR / "backend" / "skills"
PROMPTS_DIR: Path = ROOT_DIR / "backend" / "prompts"

# ── Validation ───────────────────────────────────────────────────────────────
def validate_config() -> None:
    """Raise early if critical env vars are missing."""
    missing: list[str] = []
    if not AZURE_OPENAI_API_KEY:
        missing.append("AZURE_OPENAI_KEY")
    if not AZURE_OPENAI_ENDPOINT:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not SARVAM_API_KEY:
        missing.append("SARVAM_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
