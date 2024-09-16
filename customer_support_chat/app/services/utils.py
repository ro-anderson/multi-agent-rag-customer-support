import os
import shutil
import sqlite3
from datetime import datetime
import pandas as pd
import requests
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.core.logger import logger
from qdrant_client import QdrantClient
from customer_support_chat.app.core.settings import get_settings
from typing import List, Dict

settings = get_settings()

def download_and_prepare_db():
    settings = get_settings()
    db_file = settings.SQLITE_DB_PATH
    db_dir = os.path.dirname(db_file)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
    if not os.path.exists(db_file):
        response = requests.get(db_url)
        response.raise_for_status()
        with open(db_file, "wb") as f:
            f.write(response.content)
        update_dates(db_file)

def update_dates(db_file):
    backup_file = db_file + '.backup'
    if not os.path.exists(backup_file):
        shutil.copy(db_file, backup_file)

    conn = sqlite3.connect(db_file)

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

    example_time = pd.to_datetime(
        tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
    ).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    tdf["bookings"]["book_date"] = (
        pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
        + time_diff
    )

    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    for column in datetime_columns:
        tdf["flights"][column] = (
            pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()

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

def get_qdrant_client():
    settings = get_settings()
    try:
        client = QdrantClient(url=settings.QDRANT_URL)
        # Test the connection
        client.get_collections()
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant server at {settings.QDRANT_URL}. Error: {str(e)}")
        raise

def flight_info_to_string(flight_info: List[Dict]) -> str:
    info_lines = []
    for flight in flight_info:
        line = (
            f"Flight {flight['flight_no']} from {flight['departure_airport']} "
            f"to {flight['arrival_airport']} departs at {flight['scheduled_departure']} "
            f"and arrives at {flight['scheduled_arrival']}. Seat: {flight['seat_no']}. "
            f"Fare conditions: {flight['fare_conditions']}."
        )
        info_lines.append(line)
    return "\n".join(info_lines)