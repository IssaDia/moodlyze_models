from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv
import csv
from datetime import datetime

load_dotenv()

MONGODB_URL = os.getenv('MONGODB_URL')
DB_NAME = os.getenv('DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

def load_data_from_mongodb(limit=None, batch_size=1500):
    client = MongoClient(MONGODB_URL)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    try:
        cursor = collection.find().batch_size(batch_size)
        if limit:
            cursor = cursor.limit(limit)

        for document in cursor:
            yield document
    finally:
        client.close()
