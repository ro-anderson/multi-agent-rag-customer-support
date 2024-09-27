from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools.token_info import get_summary
from customer_support_chat.app.services.assistants.assistant_base import Assistant, CompleteOrEscalate, llm
import yaml
from pathlib import Path

# Load the system prompt from the YAML file
prompt_path = Path(__file__).parent / "prompts" / "sp_token_info_assistant.yml"
with open(prompt_path, "r") as f:
    prompt_data = yaml.safe_load(f)

# Token info assistant prompt
token_info_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt_data["system_prompt"],
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# Token info tools
token_info_safe_tools = [get_summary]
token_info_sensitive_tools = []
token_info_tools = token_info_safe_tools + token_info_sensitive_tools

# Create the token info assistant runnable
token_info_runnable = token_info_prompt | llm.bind_tools(
    token_info_tools + [CompleteOrEscalate]
)

# Instantiate the token info assistant
token_info_assistant = Assistant(token_info_runnable)