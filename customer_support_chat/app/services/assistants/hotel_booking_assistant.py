from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools import (
    search_hotels,
    book_hotel,
    update_hotel,
    cancel_hotel,
)
from customer_support_chat.app.services.assistants.assistant_base import Assistant, CompleteOrEscalate, llm
import yaml
from pathlib import Path

# Load the system prompt from the YAML file
prompt_path = Path(__file__).parent / "prompts" / "sp_hotel_booking_assistant.yml"
with open(prompt_path, "r") as f:
    prompt_data = yaml.safe_load(f)

# Hotel booking assistant prompt
hotel_booking_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt_data["system_prompt"],
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# Hotel booking tools
book_hotel_safe_tools = [search_hotels]
book_hotel_sensitive_tools = [book_hotel, update_hotel, cancel_hotel]
book_hotel_tools = book_hotel_safe_tools + book_hotel_sensitive_tools

# Create the hotel booking assistant runnable
book_hotel_runnable = hotel_booking_prompt | llm.bind_tools(
    book_hotel_tools + [CompleteOrEscalate]
)

# Instantiate the hotel booking assistant
hotel_booking_assistant = Assistant(book_hotel_runnable)