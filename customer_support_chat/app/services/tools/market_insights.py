from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
import requests
from typing import List, Dict
import requests

settings = get_settings()

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