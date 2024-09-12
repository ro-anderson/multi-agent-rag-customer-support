# customer_support_chat/app/core/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DATA_PATH: str = "./customer_support_chat/data"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")

def get_settings():
    return Config()
