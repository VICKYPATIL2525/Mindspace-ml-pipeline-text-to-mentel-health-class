from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from anthropic import Anthropic
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

load_dotenv()

# LangGraph state definition
class ConsentState(TypedDict):
    messages: List[BaseMessage]
    user_input: str
    consent_given: bool

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024,
    temperature=1,
)

# Initialize a second LLM for Haiku 4.5 (faster, lighter model)
llm_haiku = ChatAnthropic(
    model="claude-haiku-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024,
    temperature=1,
)

# Initialize a third LLM using Azure OpenAI GPT-4o mini
llm_azure = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    max_tokens=1024,
    temperature=1,
)