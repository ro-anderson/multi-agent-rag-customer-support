from os import environ
from dotenv import load_dotenv


load_dotenv()

class Config:
    OPENAI_API_KEY: str = environ.get("OPENAI_API_KEY")
    SQLITE_DB_PATH: str = environ.get("SQLITE_DB_PATH", "./customer_support_chat/data/travel2.sqlite")
    QDRANT_URL: str = environ.get("QDRANT_URL", "http://localhost:6333")

def get_settings():
    return Config()
