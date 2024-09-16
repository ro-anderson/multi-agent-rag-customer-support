# customer_support_chat/app/graph.py

from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langchain_core.runnables import RunnableConfig
from customer_support_chat.app.services.assistant import (
    Assistant,
    part_2_assistant_runnable,
    part_2_tools,
)
from customer_support_chat.app.services.utils import (
    create_tool_node_with_fallback,
    flight_info_to_string,
)
from customer_support_chat.app.services.tools.flights import fetch_user_flight_information
from customer_support_chat.app.core.state import State  # Import State from state.py

def user_info(state: State, config: RunnableConfig):
    # Get passenger_id from config
    passenger_id = config.get("configurable", {}).get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    # Call fetch_user_flight_information with config
    flight_info = fetch_user_flight_information.invoke(input={}, config=config)

    
    # Convert flight_info (List[Dict]) to string
    user_info_str = flight_info_to_string(flight_info)
    
    return {"user_info": user_info_str}

# Build the graph
builder = StateGraph(State)

# Add nodes and edges (without 'requires_config' arguments)
builder.add_node("fetch_user_info", user_info)
builder.add_node("assistant", Assistant(part_2_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_2_tools))
builder.add_edge(START, "fetch_user_info")
builder.add_edge("fetch_user_info", "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# Initialize the checkpointer
memory = MemorySaver()

# Compile the graph with interrupt_before
part_2_graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"],
)
