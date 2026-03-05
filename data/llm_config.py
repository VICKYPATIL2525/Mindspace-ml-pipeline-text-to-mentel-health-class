import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI

# Load .env from project root (one level up from data/)
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


def get_llms():
    """
    Initialize and return all LLM models used for dataset generation.
    """

    # Claude Sonnet (strong reasoning model)
    llm_sonnet = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=4000,
        temperature=0.9,
    )

    # Claude Haiku (faster model)
    llm_haiku = ChatAnthropic(
        model="claude-haiku-4-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=4000,
        temperature=0.9,
    )

    # Azure OpenAI GPT-4.1 Mini
    llm_azure = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        max_tokens=4000,
        temperature=0.9,
    )

    models = {
        "claude-sonnet": llm_sonnet,
        "claude-haiku": llm_haiku,
        "gpt4o-mini": llm_azure,
    }

    return models