from vectorizer.app.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

settings = get_settings()
faq_vectordb = VectorDB(table_name="faq", collection_name="faq_collection")

@tool
def search_faq(
    query: str,
    limit: int = 2,
) -> List[Dict]:
    """Search for FAQ entries based on a natural language query."""
    search_results = faq_vectordb.search(query, limit=limit)

    faq_entries = []
    for result in search_results:
        payload = result.payload
        faq_entries.append({
            "question": payload["question"],
            "answer": payload["answer"],
            "category": payload["category"],
            "chunk": payload["content"],
            "similarity": result.score,
        })
    return faq_entries

@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted.
    Use this before making any flight changes or performing other 'write' events."""
    faq_results = search_faq(query, limit=2)
    if not faq_results:
        return "Sorry, I couldn't find any relevant policy information. Please contact support for assistance."
    
    policy_info = "\n\n".join([f"Q: {entry['question']}\nA: {entry['answer']}" for entry in faq_results])
    return f"Here's the relevant policy information:\n\n{policy_info}"
