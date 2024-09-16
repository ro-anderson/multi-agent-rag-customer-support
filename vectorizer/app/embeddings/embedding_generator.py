from openai import OpenAI
from vectorizer.app.core.settings import get_settings
from typing import Union, List

settings = get_settings()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

def generate_embedding(content: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    if isinstance(content, str):
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[content]
        )
        return response.data[0].embedding
    elif isinstance(content, list):
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=content
        )
        return [item.embedding for item in response.data]
    else:
        raise ValueError("Content must be either a string or a list of strings")
