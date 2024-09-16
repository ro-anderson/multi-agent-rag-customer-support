# customer_support_chat/app/services/assistant.py
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.services.tools import *
from customer_support_chat.app.core.state import State 
from langchain_core.runnables import RunnableConfig

settings = get_settings()

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state, config)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4",
    openai_api_key=settings.OPENAI_API_KEY,
    temperature=1,
)

# Assistant prompt template
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            "Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            "When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# List of tools

# Safe (read-only) tools
part_3_safe_tools = [
    DuckDuckGoSearchResults(max_results=10),
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    search_car_rentals,
    search_hotels,
    search_trip_recommendations,
]

# Sensitive (data-modifying) tools
part_3_sensitive_tools = [
    update_ticket_to_new_flight,
    cancel_ticket,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    book_hotel,
    update_hotel,
    cancel_hotel,
    book_excursion,
    update_excursion,
    cancel_excursion,
]


# Combine all tools
all_tools = part_3_safe_tools + part_3_sensitive_tools

# Create a set of sensitive tool names
sensitive_tool_names = {t.name for t in part_3_sensitive_tools}

# Bind the tools to the assistant
part_3_assistant_runnable = primary_assistant_prompt | llm.bind_tools(all_tools)

# Bind the tools to the assistant
part_3_assistant_runnable = primary_assistant_prompt | llm.bind_tools(all_tools)