# customer_support_chat/app/services/tools/cars.py

import sqlite3
from datetime import datetime, date
from typing import Optional, Union, List, Dict
from langchain_core.tools import tool
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.services.utils import get_qdrant_client
from customer_support_chat.app.services.vectordb.chunkenizer import recursive_character_splitting
from openai import OpenAI
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid

settings = get_settings()
db = settings.SQLITE_DB_PATH
client = OpenAI(api_key=settings.OPENAI_API_KEY)

qdrant_client = get_qdrant_client()
cars_collection = "car_rentals_collection"

def create_and_index_cars_collection():
    try:
        qdrant_client.get_collection(collection_name=cars_collection)
        print(f"Collection '{cars_collection}' already exists.")
    except Exception:
        qdrant_client.create_collection(
            collection_name=cars_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print(f"Created new collection '{cars_collection}'.")
    index_cars_data()

def index_cars_data():
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM car_rentals LIMIT 10")
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]

    points = []
    for row in rows:
        car_data = dict(zip(column_names, row))
        content = f"Car rental {car_data['name']} located at {car_data['location']}, price tier {car_data['price_tier']}."
        chunks = recursive_character_splitting(content)
        for i, chunk in enumerate(chunks):
            embedding = client.embeddings.create(
                model="text-embedding-ada-002", input=chunk
            ).data[0].embedding
            point_id = str(uuid.uuid4())
            payload = {**car_data, "chunk": chunk}
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

    qdrant_client.upsert(collection_name=cars_collection, points=points)
    conn.close()
    print("Indexed car rentals data into Qdrant.")

# Initialize collection
create_and_index_cars_collection()

@tool
def search_car_rentals(
    query: str,
    limit: int = 10,
) -> List[Dict]:
    """Search for car rentals based on a natural language query."""
    query_embedding = client.embeddings.create(
        model="text-embedding-ada-002", input=query
    ).data[0].embedding

    search_results = qdrant_client.search(
        collection_name=cars_collection,
        query_vector=query_embedding,
        limit=limit,
        with_payload=True,
    )

    rentals = []
    for result in search_results:
        payload = result.payload
        rentals.append({
            "id": payload["id"],
            "name": payload["name"],
            "location": payload["location"],
            "price_tier": payload["price_tier"],
            "chunk": payload["chunk"],
            "similarity": result.score,
        })
    return rentals

@tool
def book_car_rental(rental_id: int) -> str:
    """Book a car rental by its ID."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE car_rentals SET booked = 1 WHERE id = ?", (rental_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Car rental {rental_id} successfully booked."
    else:
        conn.close()
        return f"No car rental found with ID {rental_id}."

@tool
def update_car_rental(
    rental_id: int,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
) -> str:
    """Update a car rental's start and end dates by its ID."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    if start_date:
        cursor.execute(
            "UPDATE car_rentals SET start_date = ? WHERE id = ?",
            (start_date.strftime('%Y-%m-%d'), rental_id),
        )
    if end_date:
        cursor.execute(
            "UPDATE car_rentals SET end_date = ? WHERE id = ?",
            (end_date.strftime('%Y-%m-%d'), rental_id),
        )

    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Car rental {rental_id} successfully updated."
    else:
        conn.close()
        return f"No car rental found with ID {rental_id}."

@tool
def cancel_car_rental(rental_id: int) -> str:
    """Cancel a car rental by its ID."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE car_rentals SET booked = 0 WHERE id = ?", (rental_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Car rental {rental_id} successfully cancelled."
    else:
        conn.close()
        return f"No car rental found with ID {rental_id}."
