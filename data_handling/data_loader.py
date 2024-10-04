from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv


load_dotenv()

MONGODB_URL = os.getenv('MONGODB_URL')
DB_NAME = os.getenv('DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

print(f"MONGODB_URL: {MONGODB_URL}", f"DB_NAME: {DB_NAME}", f"COLLECTION_NAME: {COLLECTION_NAME}")

def load_data_from_mongodb():
    client = MongoClient(MONGODB_URL)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Récupération des données
    data = list(collection.find())
    
    # Vérification du nombre de documents récupérés
    print(f"Nombre de documents récupérés : {len(data)}")
    
    # Si des données sont récupérées, afficher un aperçu
    if data:
        df = pd.DataFrame(data)
        print("Aperçu des données récupérées :")
        print(df.head())  # Afficher les 5 premières lignes du DataFrame
    else:
        print("Aucune donnée trouvée dans la collection.")
    
    return pd.DataFrame(data)

# Tester la fonction
if __name__ == "__main__":
    load_data_from_mongodb()
