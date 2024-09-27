from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools.social_engagement import (
    get_engagement_sentiment_price,
    fetch_user_flight_information
)
from customer_support_chat.app.services.assistants.assistant_base import Assistant, CompleteOrEscalate, llm
import yaml
from pathlib import Path

# Load the system prompt from the YAML file
prompt_path = Path(__file__).parent / "prompts" / "sp_social_engagement_assistant.yml"
with open(prompt_path, "r") as f:
    prompt_data = yaml.safe_load(f)

# Social and Engagement assistant prompt
social_engagement_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt_data["system_prompt"],
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# Social and Engagement tools
social_engagement_safe_tools = [get_engagement_sentiment_price, fetch_user_flight_information]
social_engagement_sensitive_tools = []
social_engagement_tools = social_engagement_safe_tools + social_engagement_sensitive_tools

# Create the social and engagement assistant runnable
social_engagement_runnable = social_engagement_prompt | llm.bind_tools(
    social_engagement_tools + [CompleteOrEscalate]
)

# Instantiate the social and engagement assistant
social_engagement_assistant = Assistant(social_engagement_runnable)