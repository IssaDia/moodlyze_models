import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from data_handling.data_loader import load_data_from_mongodb
from data_handling.data_preprocessor import prepare_data_for_training
import os

MODEL_PATH = os.path.join("models", "naive_bayes.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")

def evaluate_model():
    # Load model and vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # Load and preprocess data
    raw_data = load_data_from_mongodb()
    X, y = prepare_data_for_training(raw_data)

    # Transform the test data
    X_counts = vectorizer.transform(X)
    
    # Predict the labels
    y_pred = model.predict(X_counts)

    # Evaluate model performance
    print("Accuracy: ", accuracy_score(y, y_pred))
    print("Classification Report:\n", classification_report(y, y_pred))

if __name__ == "__main__":
    evaluate_model()
