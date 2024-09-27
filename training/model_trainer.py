from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from models.data_handling.data_loader import load_data_from_mongodb
from models.data_handling.data_preprocessor import prepare_data_for_training

def train_and_evaluate_model(collection_name, db_name, connection_string):
    df = load_data_from_mongodb(collection_name, db_name, connection_string)
   
    
    X_train, X_test, y_train, y_test, vectorizer = prepare_data_for_training(df)

    
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    return model, vectorizer
