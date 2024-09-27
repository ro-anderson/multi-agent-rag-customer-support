from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools import (
    search_flights,
    lookup_policy,
)
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from customer_support_chat.app.services.assistants.assistant_base import Assistant, llm
from customer_support_chat.app.core.state import State
from pydantic import BaseModel, Field
import yaml
from pathlib import Path

# Define task delegation tools
class ToFlightBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight updates and cancellations."""
    request: str = Field(description="Any necessary follow-up questions the update flight assistant should clarify before proceeding.")

class ToBookCarRental(BaseModel):
    """Transfers work to a specialized assistant to handle car rental bookings."""
    location: str = Field(description="The location where the user wants to rent a car.")
    start_date: str = Field(description="The start date of the car rental.")
    end_date: str = Field(description="The end date of the car rental.")
    request: str = Field(description="Any additional information or requests from the user regarding the car rental.")

class ToHotelBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle hotel bookings."""
    location: str = Field(description="The location where the user wants to book a hotel.")
    checkin_date: str = Field(description="The check-in date for the hotel.")
    checkout_date: str = Field(description="The check-out date for the hotel.")
    request: str = Field(description="Any additional information or requests from the user regarding the hotel booking.")

class ToBookExcursion(BaseModel):
    """Transfers work to a specialized assistant to handle trip recommendation and other excursion bookings."""
    location: str = Field(description="The location where the user wants to book a recommended trip.")
    request: str = Field(description="Any additional information or requests from the user regarding the trip recommendation.")

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
    search_flights,
    lookup_policy,
    ToFlightBookingAssistant,
    ToBookCarRental,
    ToHotelBookingAssistant,
    ToBookExcursion,
]

# Create the primary assistant runnable
primary_assistant_runnable = primary_assistant_prompt | llm.bind_tools(primary_assistant_tools)

# Instantiate the primary assistant
primary_assistant = Assistant(primary_assistant_runnable)