from vectorizer.app.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
import sqlite3
from typing import Optional, List, Dict

settings = get_settings()
db = settings.SQLITE_DB_PATH
excursions_vectordb = VectorDB(table_name="trip_recommendations", collection_name="excursions_collection")

@tool
def search_trip_recommendations(
    query: str,
    limit: int = 2,
) -> List[Dict]:
    """Search for trip recommendations based on a natural language query."""
    search_results = excursions_vectordb.search(query, limit=limit)

    recommendations = []
    for result in search_results:
        payload = result.payload
        recommendations.append({
            "id": payload["id"],
            "name": payload["name"],
            "location": payload["location"],
            "keywords": payload["keywords"],
            "details": payload["details"],
            "booked": payload["booked"],
            "chunk": payload["content"],
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
