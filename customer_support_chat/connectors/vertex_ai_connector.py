import vertexai
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from customer_support_chat.app.core.settings import get_settings

class VertexAIConnector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VertexAIConnector, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        settings = get_settings()
        credentials_type = settings.GOOGLE_CREDENTIALS_TYPE
        credentials_data = settings.GOOGLE_APPLICATION_CREDENTIALS[credentials_type]

        if credentials_type == "authorized_user":
            required_fields = ['refresh_token', 'client_id', 'client_secret']
            if all(field in credentials_data for field in required_fields):
                self.credentials = Credentials(
                    token=None,
                    refresh_token=credentials_data['refresh_token'],
                    client_id=credentials_data['client_id'],
                    client_secret=credentials_data['client_secret'],
                    token_uri="https://oauth2.googleapis.com/token",
                    quota_project_id=credentials_data.get('quota_project_id'),
                    universe_domain=credentials_data.get('universe_domain', 'googleapis.com'),
                )
                if not self.credentials.valid:
                    self.credentials.refresh(Request())
            else:
                missing_fields = [field for field in required_fields if field not in credentials_data]
                raise ValueError(f"Missing required fields for authorized_user: {', '.join(missing_fields)}")
        elif credentials_type == "service_account":
            required_fields = ['project_id', 'private_key_id', 'private_key', 'client_email', 'client_id']
            if all(field in credentials_data for field in required_fields):
                self.credentials = service_account.Credentials.from_service_account_info(
                    {
                        "type": "service_account",
                        "project_id": credentials_data['project_id'],
                        "private_key_id": credentials_data['private_key_id'],
                        "private_key": credentials_data['private_key'],
                        "client_email": credentials_data['client_email'],
                        "client_id": credentials_data['client_id'],
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                        "client_x509_cert_url": credentials_data.get('client_x509_cert_url'),
                        "universe_domain": credentials_data.get('universe_domain', 'googleapis.com')
                    }
                )
            else:
                missing_fields = [field for field in required_fields if field not in credentials_data]
                raise ValueError(f"Missing required fields for service_account: {', '.join(missing_fields)}")
        else:
            raise ValueError(f"Unsupported credentials type: {credentials_type}")

        project_id = credentials_data.get('project_id', "third-opus-411016")
        vertexai.init(project=project_id, location="us-central1", credentials=self.credentials)

    def get_credentials(self):
        return self.credentials