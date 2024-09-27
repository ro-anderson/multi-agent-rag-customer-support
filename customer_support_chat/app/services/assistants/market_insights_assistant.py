from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools.market_insights import (
    search_market_insights,
    get_news,
)
from customer_support_chat.app.services.assistants.assistant_base import Assistant, CompleteOrEscalate, llm
import yaml
from pathlib import Path

# Load the system prompt from the YAML file
prompt_path = Path(__file__).parent / "prompts" / "sp_market_insights_assistant.yml"
with open(prompt_path, "r") as f:
    prompt_data = yaml.safe_load(f)

# Market Insights assistant prompt
market_insights_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt_data["system_prompt"],
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# Market Insights tools
market_insights_safe_tools = [search_market_insights, get_news]
market_insights_sensitive_tools = []
market_insights_tools = market_insights_safe_tools + market_insights_sensitive_tools

# Create the market insights assistant runnable
market_insights_runnable = market_insights_prompt | llm.bind_tools(
    market_insights_tools + [CompleteOrEscalate]
)

# Instantiate the market insights assistant
market_insights_assistant = Assistant(market_insights_runnable)