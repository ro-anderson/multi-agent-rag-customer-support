import uuid
from customer_support_chat.app.graph import part_1_graph, builder, memory
from customer_support_chat.app.services.utils import download_and_prepare_db
from customer_support_chat.app.core.logger import logger
from IPython.display import Image, display
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
        graph_image = part_1_graph.get_graph().draw_mermaid_png()
        graphs_dir = "./graphs"
        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)
        image_path = os.path.join(graphs_dir, "customer_support_chat_graph.png")
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
            events = part_1_graph.stream(
                {"messages": [("user", user_input)]}, config, stream_mode="values"
            )

            # Variable to track printed message IDs to avoid duplicates
            printed_message_ids = set()

            for event in events:
                message = event.get("messages")
                if message:
                    if isinstance(message, list):
                        # Get the last message (assistant's response)
                        message = message[-1]
                    if message.id not in printed_message_ids:
                        # Use pretty_print to display the assistant's response
                        message.pretty_print()
                        printed_message_ids.add(message.id)

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
