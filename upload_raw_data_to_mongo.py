from google.cloud import storage
import json
from pymongo import MongoClient
import os
import dotenv

dotenv.load_dotenv()

# GCP bucket and file details
BUCKET_NAME = os.getenv("BUCKET_NAME")
JSON_FILE_PATH = os.getenv("JSON_FILE_PATH")

# MongoDB connection details
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


def download_json_from_gcs(bucket_name, json_file_path):
    """Downloads a JSON file from GCS and returns it as a dictionary."""
    client = storage.Client()  # Assumes application default credentials
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(json_file_path)

    json_content = blob.download_as_text()
    return json.loads(json_content)


def insert_into_mongo(data, mongo_uri, db_name, collection_name):
    """Inserts the given data into MongoDB."""
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    if isinstance(data, list):
        collection.insert_many(data)
    else:
        collection.insert_one(data)

    print("Data inserted successfully!")


def main():
    try:
        json_data = download_json_from_gcs(BUCKET_NAME, JSON_FILE_PATH)

        insert_into_mongo(json_data, MONGO_URI, DB_NAME, COLLECTION_NAME)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
