from vectorizer.app.core.logger import logger
from vectorizer.app.vectordb.vectordb import VectorDB
from vectorizer.app.core.settings import get_settings

settings = get_settings()

def create_collections():
    collections = [
        ("car_rentals", "car_rentals_collection"),
        ("trip_recommendations", "excursions_collection"),
        ("flights", "flights_collection"),
        ("hotels", "hotels_collection"),
        ("faq", "faq_collection")
    ]

    for table_name, collection_name in collections:
        try:
            logger.info(f"Starting the vector database service for {table_name}")
            vectordb = VectorDB(table_name=table_name, collection_name=collection_name, create_collection=True)
            vectordb.create_embeddings()
            logger.info(f"Embedding generation and storage completed for {collection_name}")
        except Exception as e:
            logger.error(f"An error occurred while processing {table_name}: {str(e)}")
            logger.exception("Detailed error information:")

if __name__ == "__main__":
    create_collections()
