from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import os
import sys
import joblib
import pandas as pd  # Assurez-vous d'importer pandas si ce n'est pas déjà fait
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_handling.data_loader import load_data_from_mongodb
from data_handling.data_preprocessor import prepare_data_for_training

MODEL_DIR = os.path.join("..", "models/saved_models/logistic_regression")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")

def train_logistic_regression():
    
    raw_data = pd.DataFrame(list(load_data_from_mongodb()))  # Charger les données dans un DataFrame
    print(f"Nombre d'éléments dans raw_data : {len(raw_data)}")

    # Vérifiez que les colonnes existent
    if 'text' not in raw_data or 'sentiment' not in raw_data:
        print("Erreur : les colonnes 'text' ou 'sentiment' sont manquantes dans les données.")
        return

    # Extraire X et y à partir des colonnes appropriées
    X = raw_data['cleaned_text']  # Utilisez 'cleaned_text' si vous avez besoin de prétraitement
    y = raw_data['sentiment']  # Utilisez la colonne 'sentiment' pour les étiquettes

    # Préparation des données pour l'entraînement
    X_train, X_test, y_train, y_test, vectorizer = prepare_data_for_training(X, y)
    
    print(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]}")
    print(f"Taille de l'ensemble de test : {X_test.shape[0]}")
    
    # Définir les paramètres pour la recherche par grille
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'class_weight': ['balanced', None],
        'max_iter': [1000, 2000]
    }
    
    # Initialiser le modèle de base
    base_model = LogisticRegression(random_state=42)
    
    # Effectuer une recherche par grille avec validation croisée
    grid_search = GridSearchCV(base_model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Obtenir le meilleur modèle
    model = grid_search.best_estimator_
    
    accuracy = model.score(X_test, y_test)
    print(f"Précision du modèle : {accuracy:.4f}")
    
    y_pred = model.predict(X_test)
    X_test_original = vectorizer.inverse_transform(X_test.toarray())
    
    print(classification_report(y_test, y_pred))
    
    print("\nComparaison des résultats sur quelques exemples de l'ensemble de test :")
    for i in range(10):
        print(f"Tweet {i+1}:")
        print(f"  - Texte : {' '.join(X_test_original[i])}")  # Reconstruire la phrase originale
        print(f"  - Sentiment réel : {y_test.iloc[i]}")  # Utiliser .iloc pour accéder par position
        print(f"  - Sentiment prédit : {y_pred[i]}")
        print()
    
    # Sauvegarder le modèle et le vectoriseur
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, os.path.join("../models/saved_models/logistic_regression", "vectorizer_lr.pkl"))
    
    print("Modèle de régression logistique entraîné et sauvegardé avec succès.")

if __name__ == "__main__":
    train_logistic_regression()
