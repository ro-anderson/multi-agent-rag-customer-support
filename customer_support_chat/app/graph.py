from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langchain_core.runnables import RunnableConfig

from customer_support_chat.app.core.state import State
from customer_support_chat.app.services.utils import (
  create_tool_node_with_fallback,
  create_entry_node,
  token_info_to_string
)
from customer_support_chat.app.services.tools.social_engagement import fetch_user_token_information
from customer_support_chat.app.services.assistants.assistant_base import (
  CompleteOrEscalate,
  llm,
)
from customer_support_chat.app.services.assistants.primary_assistant import (
  primary_assistant,
  primary_assistant_tools,
  ToSocialEngagementAssistant,
  ToTokenInfoAssistant,
  ToMarketInsightsAssistant,
  ToSentimentAnalysisAssistant,
)
from customer_support_chat.app.services.assistants.social_engagement_assistant import (
  social_engagement_assistant,
  social_engagement_safe_tools,
  social_engagement_sensitive_tools,
)
from customer_support_chat.app.services.assistants.token_info_assistant import (
  token_info_assistant,
  token_info_safe_tools,
  token_info_sensitive_tools,
)
from customer_support_chat.app.services.assistants.market_insights_assistant import (
  market_insights_assistant,
  market_insights_safe_tools,
  market_insights_sensitive_tools,
)
from customer_support_chat.app.services.assistants.sentiment_analysis_assistant import (
  sentiment_analysis_assistant,
  sentiment_analysis_safe_tools,
  sentiment_analysis_sensitive_tools,
)

# Initialize the graph
builder = StateGraph(State)

def user_info(state: State, config: RunnableConfig):
  # Fetch user token information
  user_tokens = fetch_user_token_information(config=config)
  user_info_str = token_info_to_string(user_tokens)
  return {"user_info": user_info_str}

builder.add_node("fetch_user_info", user_info)
builder.add_edge(START, "fetch_user_info")

# Social and Engagement Assistant
builder.add_node(
  "enter_social_engagement",
  create_entry_node("Social and Engagement Assistant", "social_engagement"),
)
builder.add_node("social_engagement", social_engagement_assistant)
builder.add_edge("enter_social_engagement", "social_engagement")
builder.add_node(
  "social_engagement_safe_tools",
  create_tool_node_with_fallback(social_engagement_safe_tools),
)
builder.add_node(
  "social_engagement_sensitive_tools",
  create_tool_node_with_fallback(social_engagement_sensitive_tools),
)

def route_social_engagement(state: State) -> Literal[
  "social_engagement_safe_tools",
  "social_engagement_sensitive_tools",
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
  safe_toolnames = [t.name for t in social_engagement_safe_tools]
  if all(tc["name"] in safe_toolnames for tc in tool_calls):
      return "social_engagement_safe_tools"
  return "social_engagement_sensitive_tools"

builder.add_edge("social_engagement_safe_tools", "social_engagement")
builder.add_edge("social_engagement_sensitive_tools", "social_engagement")
builder.add_conditional_edges("social_engagement", route_social_engagement)

# Token Info Assistant
builder.add_node(
  "enter_token_info",
  create_entry_node("Token Info Assistant", "token_info"),
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

# Sentiment Analysis Assistant
builder.add_node(
  "enter_sentiment_analysis",
  create_entry_node("Sentiment Analysis Assistant", "sentiment_analysis"),
)
builder.add_node("sentiment_analysis", sentiment_analysis_assistant)
builder.add_edge("enter_sentiment_analysis", "sentiment_analysis")
builder.add_node(
  "sentiment_analysis_safe_tools",
  create_tool_node_with_fallback(sentiment_analysis_safe_tools),
)
builder.add_node(
  "sentiment_analysis_sensitive_tools",
  create_tool_node_with_fallback(sentiment_analysis_sensitive_tools),
)

def route_sentiment_analysis(state: State) -> Literal[
  "sentiment_analysis_safe_tools",
  "sentiment_analysis_sensitive_tools",
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
  safe_toolnames = [t.name for t in sentiment_analysis_safe_tools]
  if all(tc["name"] in safe_toolnames for tc in tool_calls):
      return "sentiment_analysis_safe_tools"
  return "sentiment_analysis_sensitive_tools"

builder.add_edge("sentiment_analysis_safe_tools", "sentiment_analysis")
builder.add_edge("sentiment_analysis_sensitive_tools", "sentiment_analysis")
builder.add_conditional_edges("sentiment_analysis", route_sentiment_analysis)

# Primary Assistant
builder.add_node("primary_assistant", primary_assistant)
builder.add_node(
  "primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools)
)
builder.add_edge("fetch_user_info", "primary_assistant")

def route_primary_assistant(state: State) -> Literal[
  "primary_assistant_tools",
  "enter_social_engagement",
  "enter_token_info",
  "enter_market_insights",
  "enter_sentiment_analysis",
  "__end__",
]:
  route = tools_condition(state)
  if route == END:
      return END
  tool_calls = state["messages"][-1].tool_calls
  if tool_calls:
      tool_name = tool_calls[0]["name"]
      if tool_name == ToSocialEngagementAssistant.__name__:
          return "enter_social_engagement"
      elif tool_name == ToTokenInfoAssistant.__name__:
          return "enter_token_info"
      elif tool_name == ToMarketInsightsAssistant.__name__:
          return "enter_market_insights"
      elif tool_name == ToSentimentAnalysisAssistant.__name__:
          return "enter_sentiment_analysis"
      else:
          return "primary_assistant_tools"
  return "primary_assistant"

builder.add_conditional_edges(
  "primary_assistant",
  route_primary_assistant,
  {
      "primary_assistant_tools": "primary_assistant_tools",
      "enter_social_engagement": "enter_social_engagement",
      "enter_token_info": "enter_token_info",
      "enter_market_insights": "enter_market_insights",
      "enter_sentiment_analysis": "enter_sentiment_analysis",
      END: END,
  },
)
builder.add_edge("primary_assistant_tools", "primary_assistant")

# Compile the graph with interrupts
interrupt_nodes = [
  "social_engagement_sensitive_tools",
  "token_info_sensitive_tools",
  "market_insights_sensitive_tools",
  "sentiment_analysis_sensitive_tools",
]

memory = MemorySaver()
multi_agentic_graph = builder.compile(
  checkpointer=memory,
  interrupt_before=interrupt_nodes,
)