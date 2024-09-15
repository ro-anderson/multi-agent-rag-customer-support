# customer_support_chat/app/services/assistant.py

from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.services.tools import *

settings = get_settings()

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
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

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=settings.OPENAI_API_KEY,
    temperature=1,
)

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

# Replace TavilySearchResults with DuckDuckGoSearchResults
part_1_tools = [
    DuckDuckGoSearchResults(max_results=2),
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    search_faq,  # Add this line
    update_ticket_to_new_flight,
    cancel_ticket,
    search_car_rentals,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    search_hotels,
    book_hotel,
    update_hotel,
    cancel_hotel,
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion,
]

part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)
