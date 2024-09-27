from os import environ
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account

load_dotenv()

class Config:
    OPENAI_API_KEY: str = environ.get("OPENAI_API_KEY")
    SQLITE_DB_PATH: str = environ.get("SQLITE_DB_PATH", "./customer_support_chat/data/travel2.sqlite")
    QDRANT_URL: str = environ.get("QDRANT_URL", "http://localhost:6333")
    
    # Google Application Credentials
    GOOGLE_CREDENTIALS_TYPE: str = environ.get("GOOGLE_CREDENTIALS_TYPE", "authorized_user")
    GOOGLE_APPLICATION_CREDENTIALS = {
        "authorized_user": {
            "account": environ.get("GOOGLE_ACCOUNT"),
            "client_id": environ.get("GOOGLE_CLIENT_ID"),
            "client_secret": environ.get("GOOGLE_CLIENT_SECRET"),
            "quota_project_id": environ.get("GOOGLE_QUOTA_PROJECT_ID"),
            "refresh_token": environ.get("GOOGLE_REFRESH_TOKEN"),
            "type": "authorized_user",
            "universe_domain": environ.get("GOOGLE_UNIVERSE_DOMAIN", "googleapis.com")
        },
        "service_account": {
            "type": "service_account",
            "project_id": environ.get("GOOGLE_PROJECT_ID"),
            "private_key_id": environ.get("GOOGLE_PRIVATE_KEY_ID"),
            "private_key": environ.get("GOOGLE_PRIVATE_KEY", "").replace("\\n", "\n"),
            "client_email": environ.get("GOOGLE_CLIENT_EMAIL"),
            "client_id": environ.get("GOOGLE_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": environ.get("GOOGLE_CLIENT_X509_CERT_URL"),
            "universe_domain": environ.get("GOOGLE_UNIVERSE_DOMAIN", "googleapis.com")
        }
    }

def get_settings():
    return Config()