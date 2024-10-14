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

def load_data_from_mongodb(limit=None, batch_size=1000):
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

def save_to_csv(data_generator, output_file, batch_size=1000):
    total_loaded = 0
    start_time = datetime.now()

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = None

        for batch in batch_iterator(data_generator, batch_size):
            if not writer:
                writer = csv.DictWriter(csvfile, fieldnames=batch[0].keys())
                writer.writeheader()

            writer.writerows(batch)
            total_loaded += len(batch)

            elapsed_time = (datetime.now() - start_time).total_seconds()
            print(f"Chargé {total_loaded} documents en {elapsed_time:.2f} secondes...")

    print(f"Nombre total de documents chargés : {total_loaded}")
    print(f"Temps total d'exécution : {(datetime.now() - start_time).total_seconds():.2f} secondes")

def batch_iterator(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def load_csv_to_dataframe(csv_file):
    return pd.read_csv(csv_file)

if __name__ == "__main__":
    output_file = f"{COLLECTION_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    data_generator = load_data_from_mongodb(limit=None, batch_size=1000)
    save_to_csv(data_generator, output_file, batch_size=1000)
    
    # Charger le CSV en DataFrame si nécessaire
    df = load_csv_to_dataframe(output_file)
    print(df.head())