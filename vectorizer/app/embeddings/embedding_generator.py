from typing import Union, List
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vectorizer.app.core.settings import get_settings
from customer_support_chat.connectors.vertex_ai_connector import VertexAIConnector

settings = get_settings()
vertex_ai_connector = VertexAIConnector()

def generate_embedding(content: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    credentials = vertex_ai_connector.get_credentials()
    project_id = settings.GOOGLE_APPLICATION_CREDENTIALS[settings.GOOGLE_CREDENTIALS_TYPE].get('project_id')
    
    vertexai.init(project=project_id, location="us-central1", credentials=credentials)
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    
    if isinstance(content, str):
        embeddings = model.get_embeddings([content])
        return embeddings[0].values
    elif isinstance(content, list):
        embeddings = model.get_embeddings(content)
        return [embedding.values for embedding in embeddings]
    else:
        raise ValueError("Content must be either a string or a list of strings")
