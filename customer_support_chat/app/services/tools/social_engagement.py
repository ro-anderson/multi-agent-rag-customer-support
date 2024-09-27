from vectorizer.app.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import sqlite3
from typing import Optional, Union, List, Dict
from datetime import datetime, date
import requests

settings = get_settings()
db = settings.SQLITE_DB_PATH
flights_vectordb = VectorDB(table_name="flights", collection_name="flights_collection")


@tool
def fetch_user_flight_information(*, config: RunnableConfig) -> List[Dict]:
    """Fetch all tickets for the user along with corresponding flight information and seat assignments."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = """
    SELECT 
        t.ticket_no, t.book_ref,
        f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
        bp.seat_no, tf.fare_conditions
    FROM 
        tickets t
        JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
        JOIN flights f ON tf.flight_id = f.flight_id
        LEFT JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
    WHERE 
        t.passenger_id = ?
    """
    cursor.execute(query, (passenger_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results

@tool
def get_engagement_sentiment_price(ticker: str) -> dict:
    """
    Get the engagement, sentiment and price for the previous month for a crypto coin ticker
    Each day has a sentiment score, engagement score and price, as well as the date
    This can be useful to analyse how the project has performed over the past month
    """
    try:
        req = requests.request("GET",
                               f"https://us-central1-third-opus-411016.cloudfunctions.net/SearchEngineApiV2/plot_data?query={ticker}")
        if req.status_code == 500:
            return {"description": "Token not found", 'error': "Token not found"}
        resp = req.json()
        return resp
    except Exception as e:
        print(e)
        return {"description": "Error fetching engagement, sentiment and price", "error": str(e)}