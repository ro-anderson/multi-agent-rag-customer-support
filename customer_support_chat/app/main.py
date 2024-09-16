import uuid
from customer_support_chat.app.graph import part_3_graph, memory
from customer_support_chat.app.services.utils import download_and_prepare_db
from customer_support_chat.app.core.logger import logger
from langchain_core.messages import ToolMessage
import os

def main():
    # Ensure the database is downloaded and prepared
    download_and_prepare_db()

    # Generate a unique thread ID for the session
    thread_id = str(uuid.uuid4())

    # Configuration with passenger_id and thread_id
    config = {
        "configurable": {
            "passenger_id": "8149 604011",
            "thread_id": thread_id,
        }
    }

    # Generate and save the graph visualization
    try:
        graph_image = part_3_graph.get_graph().draw_mermaid_png()
        graphs_dir = "./graphs"
        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)
        image_path = os.path.join(graphs_dir, "customer_support_chat_graph_with_sensitive_and_safe_tools.png")
        with open(image_path, "wb") as f:
            f.write(graph_image)
        print(f"Graph saved at {image_path}")
    except Exception as e:
        logger.error(f"An error occurred while generating or saving the graph: {e}")

    try:
        while True:
            user_input = input("User: ")
            if user_input.strip().lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            # Process the user input through the graph
            events = part_3_graph.stream(
                {"messages": [("user", user_input)]}, config, stream_mode="values"
            )

            # Variable to track printed message IDs to avoid duplicates
            printed_message_ids = set()

            for event in events:
                messages = event.get("messages", [])
                for message in messages:
                    if message.id not in printed_message_ids:
                        message.pretty_print()
                        printed_message_ids.add(message.id)

            # Check for interrupts
            snapshot = part_3_graph.get_state(config)
            while snapshot.next:
                # Interrupt occurred before sensitive tool execution
                user_input = input(
                    "Do you approve of the above actions? Type 'y' to continue; otherwise, explain your requested changes.\n\n"
                )
                if user_input.strip().lower() == "y":
                    # Continue execution
                    result = part_3_graph.invoke(None, config)
                else:
                    # Provide feedback to the assistant
                    tool_call_id = snapshot.value["messages"][-1].tool_calls[0]["id"]


                    result = part_3_graph.invoke(
                        {
                            "messages": [
                                ToolMessage(
                                    tool_call_id=tool_call_id,
                                    content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                                )
                            ]
                        },
                        config,
                    )
                # Process the result to display any new messages
                messages = result.get("messages", [])
                for message in messages:
                    if message.id not in printed_message_ids:
                        message.pretty_print()
                        printed_message_ids.add(message.id)
                # Update the snapshot
                snapshot = part_3_graph.get_state(config)

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()