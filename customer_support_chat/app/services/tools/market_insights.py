from vectorizer.app.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
import requests
from typing import List, Dict
import requests

settings = get_settings()
market_insights_vectordb = VectorDB(table_name="market_insights", collection_name="market_insights_collection")

@tool
def search_market_insights(
    query: str,
    limit: int = 2,
) -> List[Dict]:
    """Search for market insights based on a natural language query."""
    search_results = market_insights_vectordb.search(query, limit=limit)

    insights = []
    for result in search_results:
        payload = result.payload
        insights.append({
            "id": payload["id"],
            "title": payload["title"],
            "category": payload["category"],
            "keywords": payload["keywords"],
            "summary": payload["summary"],
            "chunk": payload["content"],
            "similarity": result.score,
        })
    return insights

@tool
def get_news() -> dict:
    """
    Get the latest news
    The news is global and is related to the crypto market
    'project' field indicates the name of the project for each article, or says 'General News' if it is not related to a specific project
    Each news article has a date it was published
    If there is a URL link to the news article, relay it to the user
    """
    try:
        req = requests.request("GET",
                               "https://us-central1-third-opus-411016.cloudfunctions.net/SearchEngineApiV2/get_news")
        resp = req.json()
        return resp
    except Exception as e:
        print(e)
        return {"description": "Error fetching news", "error": str(e)}