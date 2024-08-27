from training.model_trainer import train_and_evaluate_model
from training.model_saver import save_model
import os

from dotenv import load_dotenv

load_dotenv()



def main():
    collection_name = os.getenv('COLLECTION_NAME')
    db_name = os.getenv('DB_NAME')
    connection_string = os.getenv('MONGODB_URL')
    
    model, vectorizer = train_and_evaluate_model(collection_name, db_name, connection_string)
    save_model(model, vectorizer)

if __name__ == "__main__":
    main()
