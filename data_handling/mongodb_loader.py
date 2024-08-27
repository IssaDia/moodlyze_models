from pymongo import MongoClient
import pandas as pd
import os

def load_data_from_mongodb(collection_name, db_name='my_database', connection_string='your_mongodb_connection_string'):
    client = MongoClient(connection_string)
    db = client[db_name]
    collection = db[collection_name]
    data = list(collection.find())
    return pd.DataFrame(data)