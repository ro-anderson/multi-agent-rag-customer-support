# customer_support_chat/app/graph.py

from typing import Annotated, List, Dict, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langchain_core.runnables import RunnableConfig
from customer_support_chat.app.services.assistant import (
    Assistant,
    part_3_assistant_runnable,
    part_3_safe_tools,
    part_3_sensitive_tools,
    sensitive_tool_names,
)
from customer_support_chat.app.services.utils import (
    create_tool_node_with_fallback,
    flight_info_to_string,
)
from customer_support_chat.app.services.tools.flights import fetch_user_flight_information
from customer_support_chat.app.core.state import State  # Import State from state.py

builder = StateGraph(State)

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
builder.add_node("fetch_user_info", user_info)
builder.add_edge(START, "fetch_user_info")

builder.add_node("assistant", Assistant(part_3_assistant_runnable))
builder.add_edge("fetch_user_info", "assistant")

# Create tool nodes
builder.add_node("safe_tools", create_tool_node_with_fallback(part_3_safe_tools))
builder.add_node("sensitive_tools", create_tool_node_with_fallback(part_3_sensitive_tools))

# Define routing logic
def route_tools(state: State) -> Literal["safe_tools", "sensitive_tools", "__end__"]:
    next_node = tools_condition(state)
    if next_node == END:
        return END
    ai_message = state["messages"][-1]
    first_tool_call = ai_message.tool_calls[0]
    if first_tool_call["name"] in sensitive_tool_names:
        return "sensitive_tools"
    return "safe_tools"

builder.add_conditional_edges("assistant", route_tools)
builder.add_edge("safe_tools", "assistant")
builder.add_edge("sensitive_tools", "assistant")

# Initialize memory checkpointer
memory = MemorySaver()
part_3_graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["sensitive_tools"],
)