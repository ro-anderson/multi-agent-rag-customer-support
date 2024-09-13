import sqlite3
from typing import Optional, List, Dict
from langchain_core.tools import tool
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.services.utils import get_qdrant_client
from customer_support_chat.app.services.vectordb.chunkenizer import recursive_character_splitting
import openai
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import uuid
from tqdm import tqdm
settings = get_settings()
db = settings.SQLITE_DB_PATH
client = OpenAI(api_key=settings.OPENAI_API_KEY)

qdrant_client = get_qdrant_client()
excursions_collection = "excursions_collection"

def create_and_index_excursions_collection():
    try:
        qdrant_client.get_collection(collection_name=excursions_collection)
        print(f"Collection '{excursions_collection}' already exists.")
        if eval(settings.RECREATE_COLLECTIONS):
            print(f"Recreating collection '{excursions_collection}'.")
            qdrant_client.recreate_collection(
                collection_name=excursions_collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            index_excursions_data()
    except Exception:
        print(f"Creating new collection '{excursions_collection}'.")
        qdrant_client.create_collection(
            collection_name=excursions_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        index_excursions_data()

def index_excursions_data():
    conn = sqlite3.connect(db)
    cursor = conn.cursor() 
    cursor.execute(f"SELECT * FROM trip_recommendations LIMIT {settings.LIMIT_ROWS}")
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]

    points = []
    for row in tqdm(rows, desc="Indexing excursions"):
        excursion_data = dict(zip(column_names, row))
        content = f"Excursion {excursion_data['name']} at {excursion_data['location']} with keywords {excursion_data['keywords']}."
        chunks = recursive_character_splitting(content)
        for i, chunk in enumerate(chunks):
            embedding = client.embeddings.create(
                model="text-embedding-ada-002", input=chunk
            ).data[0].embedding
            point_id = str(uuid.uuid4())
            payload = {**excursion_data, "chunk": chunk}
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

    qdrant_client.upsert(collection_name=excursions_collection, points=points)
    conn.close()
    print("Indexed excursions data into Qdrant.")

# Initialize collection
create_and_index_excursions_collection()

@tool
def search_trip_recommendations(
    query: str,
    limit: int = 10,
) -> List[Dict]:
    """Search for trip recommendations based on a natural language query."""
    query_embedding = client.embeddings.create(
        model="text-embedding-ada-002", input=query
    ).data[0].embedding

    search_results = qdrant_client.search(
        collection_name=excursions_collection,
        query_vector=query_embedding,
        limit=limit,
        with_payload=True,
    )

    recommendations = []
    for result in search_results:
        payload = result.payload
        recommendations.append({
            "id": payload["id"],
            "name": payload["name"],
            "location": payload["location"],
            "keywords": payload["keywords"],
            "chunk": payload["chunk"],
            "similarity": result.score,
        })
    return recommendations

@tool
def book_excursion(recommendation_id: int) -> str:
    """Book an excursion by its ID."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET booked = 1 WHERE id = ?", (recommendation_id,)
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Excursion {recommendation_id} successfully booked."
    else:
        conn.close()
        return f"No excursion found with ID {recommendation_id}."

@tool
def update_excursion(recommendation_id: int, details: str) -> str:
    """Update an excursion's details by its ID."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET details = ? WHERE id = ?",
        (details, recommendation_id),
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Excursion {recommendation_id} successfully updated."
    else:
        conn.close()
        return f"No excursion found with ID {recommendation_id}."

@tool
def cancel_excursion(recommendation_id: int) -> str:
    """Cancel an excursion by its ID."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET booked = 0 WHERE id = ?", (recommendation_id,)
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Excursion {recommendation_id} successfully cancelled."
    else:
        conn.close()
        return f"No excursion found with ID {recommendation_id}."
