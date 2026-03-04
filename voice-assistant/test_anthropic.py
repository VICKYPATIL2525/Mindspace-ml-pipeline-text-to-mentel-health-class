import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize the ChatAnthropic model via LangChain (Sonnet 4.6)
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

# Define a reusable prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an AI assistant tasked with completing a specific job. "
            "Your goal is to follow the instructions carefully and produce accurate, high-quality output.\n\n"
            "Here is the task you need to complete:\n"
            "<task_description>\n{task_description}\n</task_description>\n\n"
            "Follow these guidelines:\n"
            "- Pay close attention to all details in the task description\n"
            "- Base your response solely on the information provided\n"
            "- If the task asks for a specific output format, follow it exactly\n"
            "- If you're asked to provide reasoning, provide it first, then the answer\n"
            "- Be thorough and accurate in your work\n"
            "- If the task cannot be completed with the given information, clearly explain why\n\n"
            "Provide your final response inside <response> tags."
        ),
    ),
    ("human", "{input_data}"),
])

# Build chains: one per model
chain = prompt | llm | StrOutputParser()
chain_haiku = prompt | llm_haiku | StrOutputParser()
chain_azure = prompt | llm_azure | StrOutputParser()

# --- Run the chains ---
task_description = "Answer the user's question clearly and helpfully."
input_data = "What is stress and what are 3 simple ways to manage it?"

payload = {
    "task_description": task_description,
    "input_data": input_data,
}

print("=== Sonnet 4.6 Response ===")
response = chain.invoke(payload)
print(response)

print("\n=== Haiku 4.5 Response ===")
response_haiku = chain_haiku.invoke(payload)
print(response_haiku)

print("\n=== Azure OpenAI GPT-4o Mini Response ===")
response_azure = chain_azure.invoke(payload)
print(response_azure)
