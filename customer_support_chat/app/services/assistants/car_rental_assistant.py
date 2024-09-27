from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools import (
    search_car_rentals,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
)
from customer_support_chat.app.services.assistants.assistant_base import Assistant, CompleteOrEscalate, llm
import yaml
from pathlib import Path

# Load the system prompt from the YAML file
prompt_path = Path(__file__).parent / "prompts" / "sp_car_rental_assistant.yml"
with open(prompt_path, "r") as f:
    prompt_data = yaml.safe_load(f)

# Car rental assistant prompt
car_rental_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt_data["system_prompt"],
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# Car rental tools
book_car_rental_safe_tools = [search_car_rentals]
book_car_rental_sensitive_tools = [book_car_rental, update_car_rental, cancel_car_rental]
book_car_rental_tools = book_car_rental_safe_tools + book_car_rental_sensitive_tools

# Create the car rental assistant runnable
book_car_rental_runnable = car_rental_prompt | llm.bind_tools(
    book_car_rental_tools + [CompleteOrEscalate]
)

# Instantiate the car rental assistant
car_rental_assistant = Assistant(book_car_rental_runnable)