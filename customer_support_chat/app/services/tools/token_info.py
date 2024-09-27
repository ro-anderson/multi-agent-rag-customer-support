from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
import requests

settings = get_settings()

@tool
def get_summary(ticker: str) -> dict:
    """
    Get the summary of a crypto coin ticker
    Returns a JSON object with the summary of the ticker
    Here we get the project title, description and symbol of the Crypto Coin
    We get the current price, market cap and total supply
    We get the key features and the target audience of the project, if available
    """
    try:
        req = requests.request("GET",
                               f"https://us-central1-third-opus-411016.cloudfunctions.net/SearchEngineApiV2/get_summary?query={ticker}")
        resp = req.json()
        if 'error' in resp and resp['error'] == "Invalid response from get_desc function":
            return {"description": "Token not found", 'error': "Token not found"}
        return req.json()
    except Exception as e:
        print(e)
        return {"description": "Error fetching summary", "error": str(e)}