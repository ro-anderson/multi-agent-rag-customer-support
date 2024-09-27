from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools import (
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion,
)
from customer_support_chat.app.services.assistants.assistant_base import Assistant, CompleteOrEscalate, llm
import yaml
from pathlib import Path

# Load the system prompt from the YAML file
prompt_path = Path(__file__).parent / "prompts" / "sp_excursion_assistant.yml"
with open(prompt_path, "r") as f:
    prompt_data = yaml.safe_load(f)

# Excursion assistant prompt
excursion_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt_data["system_prompt"],
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# Excursion tools
book_excursion_safe_tools = [search_trip_recommendations]
book_excursion_sensitive_tools = [book_excursion, update_excursion, cancel_excursion]
book_excursion_tools = book_excursion_safe_tools + book_excursion_sensitive_tools

# Create the excursion assistant runnable
book_excursion_runnable = excursion_prompt | llm.bind_tools(
    book_excursion_tools + [CompleteOrEscalate]
)

# Instantiate the excursion assistant
excursion_assistant = Assistant(book_excursion_runnable)