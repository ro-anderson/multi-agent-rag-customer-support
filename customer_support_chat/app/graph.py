from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from customer_support_chat.app.services.assistant import (
    Assistant,
    part_1_assistant_runnable,
    part_1_tools,
)
from customer_support_chat.app.services.utils import create_tool_node_with_fallback

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

builder = StateGraph(State)

# Define nodes
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))

# Define edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# Checkpointer
memory = MemorySaver()
part_1_graph = builder.compile(checkpointer=memory)
