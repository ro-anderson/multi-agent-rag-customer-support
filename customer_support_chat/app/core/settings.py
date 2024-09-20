from os import environ
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY: str = environ.get("OPENAI_API_KEY", "")
    DATA_PATH: str = "./customer_support_chat/data"
    LOG_LEVEL: str = environ.get("LOG_LEVEL", "DEBUG")
    SQLITE_DB_PATH: str = environ.get(
        "SQLITE_DB_PATH", "./customer_support_chat/data/travel2.sqlite"
    )
    QDRANT_URL: str = environ.get("QDRANT_URL", "http://localhost:6333")
    RECREATE_COLLECTIONS: bool = environ.get("RECREATE_COLLECTIONS", "False")
    LIMIT_ROWS: int = environ.get("LIMIT_ROWS", "100")


    # Google API Key
    GOOGLE_AI_API_KEY: str = environ.get("GOOGLE_AI_API_KEY", "")

    # API_SERVICES_URL (dockerized fastapi endpoint)
    API_SERVICES_URL = "http://127.0.0.1:8899"

    # Google Application Credentials
    GOOGLE_CREDENTIALS_TYPE: str = environ.get("GOOGLE_CREDENTIALS_TYPE", "authorized_user")
    GOOGLE_APPLICATION_CREDENTIALS = {
        # Personal credentials
        "authorized_user": {
            "account": environ.get("GOOGLE_ACCOUNT"),
            "client_id": environ.get("GOOGLE_CLIENT_ID"),
            "client_secret": environ.get("GOOGLE_CLIENT_SECRET"),
            "quota_project_id": environ.get("GOOGLE_QUOTA_PROJECT_ID"),
            "refresh_token": environ.get("GOOGLE_REFRESH_TOKEN"),
            "type": "authorized_user",
            "universe_domain": environ.get("GOOGLE_UNIVERSE_DOMAIN", "googleapis.com")
        },
        # Service account credentials
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