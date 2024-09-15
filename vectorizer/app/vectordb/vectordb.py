import os
import sqlite3
import uuid
import re
import requests
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from vectorizer.app.core.settings import get_settings
from vectorizer.app.core.logger import logger
from .chunkenizer import recursive_character_splitting
from vectorizer.app.embeddings.embedding_generator import generate_embedding

settings = get_settings()

class VectorDB:
    def __init__(self, table_name, collection_name, create_collection=False):
        self.table_name = table_name
        self.collection_name = collection_name
        self.connect_to_qdrant()
        if create_collection:
            self.create_or_clear_collection()

    def connect_to_qdrant(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)
        logger.info("Connected to Qdrant")

    def create_or_clear_collection(self):
        if self.client.collection_exists(self.collection_name):
            logger.info(f"Collection {self.collection_name} already exists. Recreating it.")
            self.client.delete_collection(collection_name=self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        else:
            logger.info(f"Creating new collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )

    def format_content(self, data, collection_name):
        # Implement formatting logic for different collections
        if collection_name == 'car_rentals_collection':
            booking_status = "booked" if data['booked'] else "not booked"
            return f"Car rental: {data['name']}, located at: {data['location']}, price tier: {data['price_tier']}. " +\
                f"Rental period starts on {data['start_date']} and ends on {data['end_date']}. " +\
                    f"Currently, the rental is: {booking_status}."

        elif collection_name == 'excursions_collection':
            booking_status = "booked" if data['booked'] else "not booked"
            return f"Excursion: {data['name']} at {data['location']}. " +\
                f"Additional details: {data['details']}. " +\
                    f"Currently, the excursion is {booking_status}. " +\
                        f"Keywords: {data['keywords']}."

        elif collection_name == 'flights_collection':

            return f"Flight {data['flight_no']} from {data['departure_airport']} to {data['arrival_airport']} " +\
                f"was scheduled to depart at {data['scheduled_departure']} and arrive at {data['scheduled_arrival']}. " +\
                    f"The actual departure was at {data['actual_departure']} and the actual arrival was at {data['actual_arrival']}. " +\
                        f"Currently, the flight status is '{data['status']}' and it was operated with aircraft code {data['aircraft_code']}."

        elif collection_name == 'hotels_collection':
            #return f"Hotel: {data['name']} in {data['location']}. Rating: {data['rating']}. Price per night: {data['price_per_night']}."
            booking_status = "booked" if data['booked'] else "not booked"
            return f"Hotel {data['name']} located in {data['location']} is categorized as {data['price_tier']} tier. " +\
                f"The check-in date is {data['checkin_date']} and the check-out date is {data['checkout_date']}. " +\
                    f"Currently, the booked status is: {booking_status}."

        elif collection_name == 'faq_collection':
            return data['page_content']  # Return the page content directly for FAQ
        else:
            return str(data)

    def create_embeddings(self):
        if self.collection_name == 'faq_collection':
            self.index_faq_docs()
        else:
            self.index_regular_docs()

    def index_regular_docs(self):
        db_connection = sqlite3.connect(settings.SQLITE_DB_PATH)
        cursor = db_connection.cursor()
        cursor.execute(f"SELECT * FROM {self.table_name} limit 3")
        rows = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]

        for row in tqdm(rows, desc=f"Indexing {self.collection_name}"):
            data = dict(zip(column_names, row))
            content = self.format_content(data, self.collection_name)
            chunks = recursive_character_splitting(content)

            for chunk in chunks:
                try:
                    embedding = generate_embedding(chunk)
                    point_id = str(uuid.uuid4())

                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=[
                            PointStruct(
                                id=point_id,
                                vector=embedding,
                                payload={
                                    "content": chunk,
                                    **data
                                }
                            )
                        ]
                    )
                    logger.info(f"Inserted point {point_id} into collection {self.collection_name}")
                except Exception as e:
                    logger.error(f"Error inserting point: {e}")
                    continue

        db_connection.close()

    def index_faq_docs(self):
        faq_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
        try:
            response = requests.get(faq_url)
            response.raise_for_status()
            faq_text = response.text

            docs = [{"page_content": txt.strip()} for txt in re.split(r"(?=\n##)", faq_text) if txt.strip()]

            points = []
            for doc in tqdm(docs, desc="Generating embeddings for FAQ documents"):
                try:
                    vector = generate_embedding(doc['page_content'])
                    points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            payload={"page_content": doc["page_content"]}
                        )
                    )
                except Exception as e:
                    logger.error(f"Error generating embedding for FAQ document: {e}")
                    continue

            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                logger.info(f"Indexed {len(points)} FAQ documents into Qdrant.")
            else:
                logger.warning("No FAQ documents were successfully embedded and indexed.")
        except Exception as e:
            logger.error(f"Error indexing FAQ documents: {e}")

    def search(self, query, limit=5, with_payload=True):
        query_vector = generate_embedding(query)
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=with_payload
        )
        return search_result

if __name__ == "__main__":
    vectordb = VectorDB("example_table", "example_collection")
    vectordb.create_embeddings()
