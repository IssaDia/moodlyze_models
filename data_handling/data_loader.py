from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv


load_dotenv()

MONGODB_URL = os.getenv('MONGODB_URL')
DB_NAME = os.getenv('DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')


def load_data_from_mongodb(limit=None, batch_size=100):
    client = MongoClient(MONGODB_URL)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    all_data = []
    total_loaded = 0
    
    # Appliquer la limite au niveau de la requête initiale si spécifiée
    query = {}
    if limit:
        cursor = collection.find(query).limit(limit)
    else:
        cursor = collection.find(query)
    
    try:
        while True:
            batch = list(cursor.batch_size(batch_size))
            if not batch:
                break
            all_data.extend(batch)
            total_loaded += len(batch)
            print(f"Chargé {total_loaded} documents...")
            
            if limit and total_loaded >= limit:
                break
    finally:
        client.close()
    
    df = pd.DataFrame(all_data)
    print(f"Nombre total de documents chargés : {len(df)}")
    return df


# Tester la fonction
if __name__ == "__main__":
    load_data_from_mongodb()
