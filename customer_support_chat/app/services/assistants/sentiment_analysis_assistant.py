from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools.sentiment_analysis import (
    get_sa
)
from customer_support_chat.app.services.assistants.assistant_base import Assistant, CompleteOrEscalate, llm
import yaml
from pathlib import Path

# Load the system prompt from the YAML file
prompt_path = Path(__file__).parent / "prompts" / "sp_sentiment_analysis_assistant.yml"
with open(prompt_path, "r") as f:
    prompt_data = yaml.safe_load(f)

# Sentiment Analysis assistant prompt
sentiment_analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt_data["system_prompt"],
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# Sentiment Analysis tools
sentiment_analysis_safe_tools = [get_sa]
sentiment_analysis_sensitive_tools = []
sentiment_analysis_tools = sentiment_analysis_safe_tools + sentiment_analysis_sensitive_tools

# Create the sentiment analysis assistant runnable
sentiment_analysis_runnable = sentiment_analysis_prompt | llm.bind_tools(
    sentiment_analysis_tools + [CompleteOrEscalate]
)

# Instantiate the sentiment analysis assistant
sentiment_analysis_assistant = Assistant(sentiment_analysis_runnable)