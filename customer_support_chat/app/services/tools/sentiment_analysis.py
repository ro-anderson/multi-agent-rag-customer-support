from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
from typing import Optional, Union, List, Dict
from datetime import datetime, date
import requests

settings = get_settings()

@tool
def get_sa(ticker: str) -> dict:
    """
    Get the sentiment analysis (SA) for a crypto coin ticker.
    This provides insights into the current market sentiment towards the token, which can be useful for making informed decisions.
    """
    try:
        req = requests.get(
            f"https://us-central1-third-opus-411016.cloudfunctions.net/SearchEngineApiV2/get_sa?query={ticker}"
        )
        if req.status_code == 500:
            return {"description": "Token not found", "error": "Token not found"}
        resp = req.json()
        return resp
    except Exception as e:
        print(e)
        return {"description": "Error fetching sentiment analysis", "error": str(e)}
