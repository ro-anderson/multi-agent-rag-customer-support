from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from typing import List
import requests

settings = get_settings()

def fetch_user_token_information(*, config: RunnableConfig) -> List[str]:
    """Fetch all tokens for the user."""
    configuration = config.get("configurable", {})
    user_email = configuration.get("user_email", None)
    if not user_email:
        raise ValueError("No user email configured.")

    try:
        response = requests.get("https://us-central1-third-opus-411016.cloudfunctions.net/SearchEngineApiV2/get_user_data")
        response.raise_for_status()
        user_data = response.json()
        return user_data.get(user_email, [])
    except requests.RequestException as e:
        print(f"Error fetching user data: {e}")
        return []

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