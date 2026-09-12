from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from prompts import RAG_WORKFLOW_INSTRUCTIONS, SUBAGENT_DELEGATION_INSTRUCTIONS, CHUNK_ANALYST_INSTRUCTIONS
from tools import search_documentation, backend
from langchain.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

max_concurrent_analysts = 3

INSTRUCTIONS = (
    RAG_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "="*80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_analysts = max_concurrent_analysts
    )
)

chunk_analyst_subagent = {
    "name": "chunk-analyst",
    "description": (
        "Analyse one retrieved documentation chunk file."
        "Pass the user's question and a single file path under /retrieved/."

    ),
    "system_prompt":CHUNK_ANALYST_INSTRUCTIONS
}


model = init_chat_model(
    model="gpt-5.6-sol",
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

agent = create_deep_agent(
    model =model,
    tools= [search_documentation],
    backend=backend,
    system_prompt=INSTRUCTIONS,
    subagents=[chunk_analyst_subagent],
)

example_query = "what are the cause of climate change"
result = agent.invoke(
    {
        "messages": [HumanMessage(content=example_query)]
    }
)

print(result["messages"][-1].content)