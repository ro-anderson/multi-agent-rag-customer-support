# customer_support_chat/app/services/tools/flights.py

import sqlite3
from datetime import datetime, date
from typing import Optional, Union, List, Dict
import pytz
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.services.utils import get_qdrant_client
from customer_support_chat.app.services.vectordb.chunkenizer import recursive_character_splitting
from openai import OpenAI
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid

settings = get_settings()
db = settings.SQLITE_DB_PATH
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Initialize Qdrant client
qdrant_client = get_qdrant_client()
flights_collection = "flights_collection"

def create_and_index_flights_collection():
    try:
        qdrant_client.get_collection(collection_name=flights_collection)
        print(f"Collection '{flights_collection}' already exists.")
    except Exception:
        qdrant_client.create_collection(
            collection_name=flights_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print(f"Created new collection '{flights_collection}'.")
    index_flights_data()

def index_flights_data():
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM flights LIMIT 10")
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]

    points = []
    for row in rows:
        flight_data = dict(zip(column_names, row))
        content = f"Flight {flight_data['flight_no']} from {flight_data['departure_airport']} to {flight_data['arrival_airport']} departs at {flight_data['scheduled_departure']} and arrives at {flight_data['scheduled_arrival']}."
        chunks = recursive_character_splitting(content)
        for i, chunk in enumerate(chunks):
            embedding = client.embeddings.create(
                model="text-embedding-ada-002", input=chunk
            ).data[0].embedding
            point_id = str(uuid.uuid4())
            payload = {**flight_data, "chunk": chunk}
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

    qdrant_client.upsert(collection_name=flights_collection, points=points)
    conn.close()
    print("Indexed flights data into Qdrant.")

# Initialize collection
create_and_index_flights_collection()

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
def search_flights(
    query: str,
    limit: int = 20,
) -> List[Dict]:
    """Search for flights based on a natural language query."""
    query_embedding = client.embeddings.create(
        model="text-embedding-ada-002", input=query
    ).data[0].embedding

    search_results = qdrant_client.search(
        collection_name=flights_collection,
        query_vector=query_embedding,
        limit=limit,
        with_payload=True,
    )

    flights = []
    for result in search_results:
        payload = result.payload
        flights.append({
            "flight_id": payload["flight_id"],
            "flight_no": payload["flight_no"],
            "departure_airport": payload["departure_airport"],
            "arrival_airport": payload["arrival_airport"],
            "scheduled_departure": payload["scheduled_departure"],
            "scheduled_arrival": payload["scheduled_arrival"],
            "chunk": payload["chunk"],
            "similarity": result.score,
        })
    return flights

@tool
def update_ticket_to_new_flight(
    ticket_no: str, new_flight_id: int, *, config: RunnableConfig
) -> str:
    """Update the user's ticket to a new valid flight."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # Check if the ticket exists and belongs to the passenger
    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    ticket = cursor.fetchone()
    if not ticket:
        conn.close()
        return f"Ticket {ticket_no} not found for passenger {passenger_id}."

    # Update the flight in ticket_flights
    cursor.execute(
        "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
        (new_flight_id, ticket_no),
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Ticket {ticket_no} successfully updated to flight {new_flight_id}."
    else:
        conn.close()
        return f"Failed to update ticket {ticket_no}."

@tool
def cancel_ticket(ticket_no: str, *, config: RunnableConfig) -> str:
    """Cancel the user's ticket and remove it from the database."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # Check if the ticket exists and belongs to the passenger
    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    ticket = cursor.fetchone()
    if not ticket:
        conn.close()
        return f"Ticket {ticket_no} not found for passenger {passenger_id}."

    # Delete from ticket_flights
    cursor.execute(
        "DELETE FROM ticket_flights WHERE ticket_no = ?",
        (ticket_no,),
    )
    # Delete from tickets
    cursor.execute(
        "DELETE FROM tickets WHERE ticket_no = ?",
        (ticket_no,),
    )
    conn.commit()

    conn.close()
    return f"Ticket {ticket_no} successfully cancelled."