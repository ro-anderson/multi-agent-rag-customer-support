from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langchain_core.runnables import RunnableConfig

from customer_support_chat.app.core.state import State
from customer_support_chat.app.services.utils import (
  create_tool_node_with_fallback,
  flight_info_to_string,
  create_entry_node,
)
from customer_support_chat.app.services.tools.flights import fetch_user_flight_information
from customer_support_chat.app.services.assistants.assistant_base import (
  Assistant,
  CompleteOrEscalate,
  llm,
)
from customer_support_chat.app.services.assistants.primary_assistant import (
  primary_assistant,
  primary_assistant_tools,
  ToFlightBookingAssistant,
  ToHotelBookingAssistant,
  ToTokenInfoAssistant,
  ToMarketInsightsAssistant,
)
from customer_support_chat.app.services.assistants.flight_booking_assistant import (
  flight_booking_assistant,
  update_flight_safe_tools,
  update_flight_sensitive_tools,
)
from customer_support_chat.app.services.assistants.token_info_assistant import (
  token_info_assistant,
  token_info_safe_tools,
  token_info_sensitive_tools,
)
from customer_support_chat.app.services.assistants.hotel_booking_assistant import (
  hotel_booking_assistant,
  book_hotel_safe_tools,
  book_hotel_sensitive_tools,
)
from customer_support_chat.app.services.assistants.market_insights_assistant import (
  market_insights_assistant,
  market_insights_safe_tools,
  market_insights_sensitive_tools,
)

# Initialize the graph
builder = StateGraph(State)

def user_info(state: State, config: RunnableConfig):
  # Fetch user flight information
  flight_info = fetch_user_flight_information.invoke(input={}, config=config)
  user_info_str = flight_info_to_string(flight_info)
  return {"user_info": user_info_str}

builder.add_node("fetch_user_info", user_info)
builder.add_edge(START, "fetch_user_info")

# Flight Booking Assistant
builder.add_node(
  "enter_update_flight",
  create_entry_node("Flight Updates & Booking Assistant", "update_flight"),
)
builder.add_node("update_flight", flight_booking_assistant)
builder.add_edge("enter_update_flight", "update_flight")
builder.add_node(
  "update_flight_safe_tools",
  create_tool_node_with_fallback(update_flight_safe_tools),
)
builder.add_node(
  "update_flight_sensitive_tools",
  create_tool_node_with_fallback(update_flight_sensitive_tools),
)

def route_update_flight(state: State) -> Literal[
  "update_flight_safe_tools",
  "update_flight_sensitive_tools",
  "primary_assistant",
  "__end__",
]:
  route = tools_condition(state)
  if route == END:
      return END
  tool_calls = state["messages"][-1].tool_calls
  did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
  if did_cancel:
      return "primary_assistant"
  safe_toolnames = [t.name for t in update_flight_safe_tools]
  if all(tc["name"] in safe_toolnames for tc in tool_calls):
      return "update_flight_safe_tools"
  return "update_flight_sensitive_tools"

builder.add_edge("update_flight_safe_tools", "update_flight")
builder.add_edge("update_flight_sensitive_tools", "update_flight")
builder.add_conditional_edges("update_flight", route_update_flight)

# Token Info Assistant
builder.add_node(
  "enter_token_info",
  create_entry_node("Token Information Assistant", "token_info"),
)
builder.add_node("token_info", token_info_assistant)
builder.add_edge("enter_token_info", "token_info")
builder.add_node(
  "token_info_safe_tools",
  create_tool_node_with_fallback(token_info_safe_tools),
)
builder.add_node(
  "token_info_sensitive_tools",
  create_tool_node_with_fallback(token_info_sensitive_tools),
)

def route_token_info(state: State) -> Literal[
  "token_info_safe_tools",
  "token_info_sensitive_tools",
  "primary_assistant",
  "__end__",
]:
  route = tools_condition(state)
  if route == END:
      return END
  tool_calls = state["messages"][-1].tool_calls
  did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
  if did_cancel:
      return "primary_assistant"
  safe_toolnames = [t.name for t in token_info_safe_tools]
  if all(tc["name"] in safe_toolnames for tc in tool_calls):
      return "token_info_safe_tools"
  return "token_info_sensitive_tools"

builder.add_edge("token_info_safe_tools", "token_info")
builder.add_edge("token_info_sensitive_tools", "token_info")
builder.add_conditional_edges("token_info", route_token_info)

# Hotel Booking Assistant
builder.add_node(
  "enter_book_hotel",
  create_entry_node("Hotel Booking Assistant", "book_hotel"),
)
builder.add_node("book_hotel", hotel_booking_assistant)
builder.add_edge("enter_book_hotel", "book_hotel")
builder.add_node(
  "book_hotel_safe_tools",
  create_tool_node_with_fallback(book_hotel_safe_tools),
)
builder.add_node(
  "book_hotel_sensitive_tools",
  create_tool_node_with_fallback(book_hotel_sensitive_tools),
)

def route_book_hotel(state: State) -> Literal[
  "book_hotel_safe_tools",
  "book_hotel_sensitive_tools",
  "primary_assistant",
  "__end__",
]:
  route = tools_condition(state)
  if route == END:
      return END
  tool_calls = state["messages"][-1].tool_calls
  did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
  if did_cancel:
      return "primary_assistant"
  safe_toolnames = [t.name for t in book_hotel_safe_tools]
  if all(tc["name"] in safe_toolnames for tc in tool_calls):
      return "book_hotel_safe_tools"
  return "book_hotel_sensitive_tools"

builder.add_edge("book_hotel_safe_tools", "book_hotel")
builder.add_edge("book_hotel_sensitive_tools", "book_hotel")
builder.add_conditional_edges("book_hotel", route_book_hotel)

# Market Insights Assistant
builder.add_node(
  "enter_market_insights",
  create_entry_node("Market Insights Assistant", "market_insights"),
)
builder.add_node("market_insights", market_insights_assistant)
builder.add_edge("enter_market_insights", "market_insights")
builder.add_node(
  "market_insights_safe_tools",
  create_tool_node_with_fallback(market_insights_safe_tools),
)
builder.add_node(
  "market_insights_sensitive_tools",
  create_tool_node_with_fallback(market_insights_sensitive_tools),
)

def route_market_insights(state: State) -> Literal[
  "market_insights_safe_tools",
  "market_insights_sensitive_tools",
  "primary_assistant",
  "__end__",
]:
  route = tools_condition(state)
  if route == END:
      return END
  tool_calls = state["messages"][-1].tool_calls
  did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
  if did_cancel:
      return "primary_assistant"
  safe_toolnames = [t.name for t in market_insights_safe_tools]
  if all(tc["name"] in safe_toolnames for tc in tool_calls):
      return "market_insights_safe_tools"
  return "market_insights_sensitive_tools"

builder.add_edge("market_insights_safe_tools", "market_insights")
builder.add_edge("market_insights_sensitive_tools", "market_insights")
builder.add_conditional_edges("market_insights", route_market_insights)

# Primary Assistant
builder.add_node("primary_assistant", primary_assistant)
builder.add_node(
  "primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools)
)
builder.add_edge("fetch_user_info", "primary_assistant")

def route_primary_assistant(state: State) -> Literal[
  "primary_assistant_tools",
  "enter_update_flight",
  "enter_token_info",
  "enter_book_hotel",
  "enter_market_insights",
  "__end__",
]:
  route = tools_condition(state)
  if route == END:
      return END
  tool_calls = state["messages"][-1].tool_calls
  if tool_calls:
      tool_name = tool_calls[0]["name"]
      if tool_name == ToFlightBookingAssistant.__name__:
          return "enter_update_flight"
      elif tool_name == ToTokenInfoAssistant.__name__:
          return "enter_token_info"
      elif tool_name == ToHotelBookingAssistant.__name__:
          return "enter_book_hotel"
      elif tool_name == ToMarketInsightsAssistant.__name__:
          return "enter_market_insights"
      else:
          return "primary_assistant_tools"
  return "primary_assistant"

builder.add_conditional_edges(
  "primary_assistant",
  route_primary_assistant,
  {
      "enter_update_flight": "enter_update_flight",
      "enter_token_info": "enter_token_info",
      "enter_book_hotel": "enter_book_hotel",
      "enter_market_insights": "enter_market_insights",
      "primary_assistant_tools": "primary_assistant_tools",
      END: END,
  },
)
builder.add_edge("primary_assistant_tools", "primary_assistant")

# Compile the graph with interrupts
interrupt_nodes = [
  "update_flight_sensitive_tools",
  "token_info_sensitive_tools",
  "book_hotel_sensitive_tools",
  "market_insights_sensitive_tools",
]

memory = MemorySaver()
multi_agentic_graph = builder.compile(
  checkpointer=memory,
  interrupt_before=interrupt_nodes,
)