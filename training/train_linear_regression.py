import sys
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_handling.data_loader import load_data_from_mongodb

def train_linear_regression():
    # Chargement des données
    df = pd.DataFrame(load_data_from_mongodb())
    data = df[["cleaned_text", "sentiment"]].dropna()

    # Encodage des sentiments
    label_encoder = LabelEncoder()
    data["sentiment_encoded"] = label_encoder.fit_transform(data["sentiment"])

    X = data["cleaned_text"]
    y = data["sentiment_encoded"]

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorisation avec CountVectorizer
    vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    # Entraînement du modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train_vect, y_train)

    # Prédictions sur le jeu de test
    y_pred = model.predict(X_test_vect)

    # Conversion des prédictions en classes pour comparaison (arrondi)
    y_pred_classes = np.rint(y_pred).astype(int)

    # Évaluation du modèle
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = np.mean(y_pred_classes == y_test)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")
    print(f"Accuracy (approximative): {accuracy:.4f}")

    # Exemple de prédiction
    example_text = "This is a test example for regression."
    example_vector = vectorizer.transform([example_text])
    example_prediction = model.predict(example_vector)[0]

    print(f"\nTexte d'exemple: {example_text}")
    print(f"Prédiction pour le texte (encodage): {example_prediction:.2f}")

if __name__ == "__main__":
    train_linear_regression()
