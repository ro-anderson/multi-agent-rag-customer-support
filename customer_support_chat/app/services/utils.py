from datetime import datetime
import pandas as pd
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.core.logger import logger
from customer_support_chat.app.core.settings import get_settings
from typing import List, Dict, Callable

from langchain_core.messages import ToolMessage
from customer_support_chat.app.core.state import State

settings = get_settings()


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=(
                        f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user. "
                        f"The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name}, "
                        "and the booking, update, or other action is not complete until after you have successfully invoked the appropriate tool. "
                        "If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control. "
                        "Do not mention who you are—just act as the proxy for the assistant."
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }
    return entry_node

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            {
                "type": "tool",
                "content": f"Error: {repr(error)}\nPlease fix your mistakes.",
                "tool_call_id": tc["id"],
            }
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list):
    from langchain_core.messages import ToolMessage
    from langchain_core.runnables import RunnableLambda
    from langgraph.prebuilt import ToolNode

    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def token_info_to_string(token_info: List[str]) -> str:
    info_lines = []
    for i, token in enumerate(token_info, 1):
        line = f"[{i}] {token}\n"
        info_lines.append(line)

    header = "User's previously searched tokens on SpectreAI:\n"
    if not info_lines:
        return header + "No tokens searched yet."
    
    return header + "".join(info_lines)