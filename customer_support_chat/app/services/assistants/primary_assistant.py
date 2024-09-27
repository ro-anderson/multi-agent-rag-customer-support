from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools import (
    lookup_policy,
)
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from customer_support_chat.app.services.assistants.assistant_base import Assistant, llm
from customer_support_chat.app.core.state import State
from pydantic import BaseModel, Field
from typing import Optional
import yaml
from pathlib import Path

# Define task delegation tools
class ToSocialEngagementAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle social engagement and performance analysis for crypto projects."""
    ticker: str = Field(description="The ticker or symbol of the cryptocurrency token.")
    request: str = Field(description="Any additional information or specific queries about the token's social engagement and performance.")

class ToTokenInfoAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle token information queries."""
    ticker: str = Field(description="The ticker or symbol of the cryptocurrency token.")
    request: str = Field(description="Any additional information or specific queries about the token.")

class ToMarketInsightsAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle market insights and news related to the crypto market."""
    query: str = Field(description="The user's query about market insights or crypto news.")
    specific_token: Optional[str] = Field(default=None, description="The specific cryptocurrency token the user is interested in, if any.")

class ToSentimentAnalysisAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle sentiment analysis related to the crypto market."""
    query: str = Field(description="The user's query about sentiment analysis or market mood.")
    specific_token: Optional[str] = Field(default=None, description="The specific cryptocurrency token the user is interested in, if any.")

# Load the system prompt from the YAML file
prompt_path = Path(__file__).parent / "prompts" / "sp_primary_assistant.yml"
with open(prompt_path, "r") as f:
    prompt_data = yaml.safe_load(f)

# Primary assistant prompt
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt_data["system_prompt"],
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# Primary assistant tools
primary_assistant_tools = [
    DuckDuckGoSearchResults(max_results=10),
    lookup_policy,
    ToSocialEngagementAssistant,
    ToTokenInfoAssistant,
    ToMarketInsightsAssistant,
    ToSentimentAnalysisAssistant,
]

# Create the primary assistant runnable
primary_assistant_runnable = primary_assistant_prompt | llm.bind_tools(primary_assistant_tools)

# Instantiate the primary assistant
primary_assistant = Assistant(primary_assistant_runnable)