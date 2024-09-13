# customer_support_chat/app/services/tools/lookup.py
import re
import numpy as np
import requests
from langchain_core.tools import tool
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.services.utils import get_qdrant_client
from openai import OpenAI
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import logging
import uuid

# Add this near the top of the file, after imports
logger = logging.getLogger(__name__)

settings = get_settings()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

faq_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
response = requests.get(faq_url)
response.raise_for_status()
faq_text = response.text

docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]

class VectorStoreRetriever:
    def __init__(self, qdrant_client, openai_client, collection_name="faq_collection"):
        self.qdrant_client = qdrant_client
        self.openai_client = openai_client
        self.collection_name = collection_name
        self.create_collection()

    def create_collection(self):
        try:
            self.qdrant_client.get_collection(collection_name=self.collection_name)
            print(f"Collection '{self.collection_name}' already exists.")
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            print(f"Created new collection '{self.collection_name}'.")
        
        # Always index docs, whether the collection is new or existing
        self.index_docs(docs)

    def index_docs(self, docs):
        vectors = self.generate_embeddings([doc["page_content"] for doc in docs])
        points = [
            PointStruct(
                id=str(uuid.uuid4()),  # Generate a UUID for each point
                vector=vector,
                payload={"page_content": doc["page_content"]},
            )
            for vector, doc in zip(vectors, docs)
        ]
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        print(f"Indexed {len(points)} documents into Qdrant.")

    def generate_embeddings(self, texts):
        embeddings = self.openai_client.embeddings.create(
            model="text-embedding-ada-002", input=texts
        )
        vectors = [data.embedding for data in embeddings.data]
        return vectors

    def query(self, query: str, k: int = 5) -> list[dict]:
        query_embedding = self.generate_embeddings([query])[0]
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            with_payload=True,
        )
        return [
            {"page_content": hit.payload["page_content"], "similarity": hit.score}
            for hit in search_result
        ]

# Initialize Qdrant client and retriever
try:
    qdrant_client = get_qdrant_client()
    retriever = VectorStoreRetriever(qdrant_client=qdrant_client, openai_client=client)
except Exception as e:
    logger.error(f"Failed to initialize VectorStoreRetriever. Error: {str(e)}")
    # You might want to provide a fallback mechanism here
    retriever = None

@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted.
    Use this before making any flight changes performing other 'write' events."""
    if retriever is None:
        return "Sorry, I'm unable to access the policy database at the moment. Please try again later or contact support."
    docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])
