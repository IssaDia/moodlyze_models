from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
from data_handling.data_loader import load_data_from_mongodb
from data_handling.data_preprocessor import prepare_data_for_training

# Path to save the trained model
MODEL_DIR = os.path.join("..", "models/saved_models/naive_bayes")
MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes.pkl")

def train_naive_bayes():
    raw_data = load_data_from_mongodb() 
    print(f"Nombre d'éléments dans raw_data : {len(raw_data)}")
    X_train, X_test, y_train, y_test, vectorizer = prepare_data_for_training(raw_data) 
    print(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]}")
    print(f"Taille de l'ensemble de test : {X_test.shape[0]}")

    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Accuracy du modèle : {accuracy:.4f}")

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    X_test_original = vectorizer.inverse_transform(X_test.toarray())

    print("\nComparaison des résultats sur quelques exemples de l'ensemble de test :")
    for i in range(10):
        print(f"Tweet {i+1}:")
        print(f"  - Texte : {' '.join(X_test_original[i])}")  # Reconstruire la phrase originale
        print(f"  - Sentiment réel : {y_test.iloc[i]}")  # Utiliser .iloc pour accéder par position
        print(f"  - Sentiment prédit : {y_pred[i]}")
        print()

    # Save the trained model and vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, os.path.join("../models/saved_models/naive_bayes", "vectorizer_nb.pkl"))

    print("Naive Bayes model trained and saved successfully.")

if __name__ == "__main__":
    train_naive_bayes()
