
import sqlite3
from datetime import datetime, date
from typing import Optional, Union, List, Dict
from langchain_core.tools import tool
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.services.utils import get_qdrant_client
from customer_support_chat.app.services.vectordb.chunkenizer import recursive_character_splitting
from openai import OpenAI
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
import uuid

settings = get_settings()
db = settings.SQLITE_DB_PATH
client = OpenAI(api_key=settings.OPENAI_API_KEY)

qdrant_client = get_qdrant_client()
hotels_collection = "hotels_collection"

def create_and_index_hotels_collection():
    try:
        qdrant_client.get_collection(collection_name=hotels_collection)
        print(f"Collection '{hotels_collection}' already exists.")
        if eval(settings.RECREATE_COLLECTIONS):
            print(f"Recreating collection '{hotels_collection}'.")
            qdrant_client.recreate_collection(
                collection_name=hotels_collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            index_hotels_data()
    except Exception:
        print(f"Creating new collection '{hotels_collection}'.")
        qdrant_client.create_collection(
            collection_name=hotels_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        index_hotels_data()

def index_hotels_data():
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM hotels LIMIT {settings.LIMIT_ROWS}")
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]

    points = []
    for row in tqdm(rows, desc="Indexing hotels"):
        hotel_data = dict(zip(column_names, row))
        content = f"Hotel {hotel_data['name']} located at {hotel_data['location']}, price tier {hotel_data['price_tier']}."
        chunks = recursive_character_splitting(content)
        for i, chunk in enumerate(chunks):
            embedding = client.embeddings.create(
                model="text-embedding-ada-002", input=chunk
            ).data[0].embedding
            point_id = str(uuid.uuid4())
            payload = {**hotel_data, "chunk": chunk}
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

    qdrant_client.upsert(collection_name=hotels_collection, points=points)
    conn.close()
    print("Indexed hotels data into Qdrant.")

# Initialize collection
create_and_index_hotels_collection()

@tool
def search_hotels(
    query: str,
    limit: int = 10,
) -> List[Dict]:
    """Search for hotels based on a natural language query."""
    query_embedding = client.embeddings.create(
        model="text-embedding-ada-002", input=query
    ).data[0].embedding

    search_results = qdrant_client.search(
        collection_name=hotels_collection,
        query_vector=query_embedding,
        limit=limit,
        with_payload=True,
    )

    hotels = []
    for result in search_results:
        payload = result.payload
        hotels.append({
            "id": payload["id"],
            "name": payload["name"],
            "location": payload["location"],
            "price_tier": payload["price_tier"],
            "chunk": payload["chunk"],
            "similarity": result.score,
        })
    return hotels

@tool
def book_hotel(hotel_id: int) -> str:
    """Book a hotel by its ID."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE hotels SET booked = 1 WHERE id = ?", (hotel_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Hotel {hotel_id} successfully booked."
    else:
        conn.close()
        return f"No hotel found with ID {hotel_id}."

@tool
def update_hotel(
    hotel_id: int,
    checkin_date: Optional[Union[datetime, date]] = None,
    checkout_date: Optional[Union[datetime, date]] = None,
) -> str:
    """Update a hotel's check-in and check-out dates by its ID."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    if checkin_date:
        cursor.execute(
            "UPDATE hotels SET checkin_date = ? WHERE id = ?",
            (checkin_date.strftime('%Y-%m-%d'), hotel_id),
        )
    if checkout_date:
        cursor.execute(
            "UPDATE hotels SET checkout_date = ? WHERE id = ?",
            (checkout_date.strftime('%Y-%m-%d'), hotel_id),
        )

    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Hotel {hotel_id} successfully updated."
    else:
        conn.close()
        return f"No hotel found with ID {hotel_id}."

@tool
def cancel_hotel(hotel_id: int) -> str:
    """Cancel a hotel by its ID."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE hotels SET booked = 0 WHERE id = ?", (hotel_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Hotel {hotel_id} successfully cancelled."
    else:
        conn.close()
        return f"No hotel found with ID {hotel_id}."
