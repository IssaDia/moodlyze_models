from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_handling.data_loader import load_data_from_mongodb
from data_handling.data_preprocessor import prepare_data_for_training

# Chemin pour sauvegarder le modèle
MODEL_DIR = os.path.join("..", "models/saved_models/logistic_regression")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")

def analyze_class_distribution(y, title="Distribution des classes"):
    """Affiche la distribution des classes dans le dataset."""
    distribution = pd.Series(y).value_counts()
    total = len(y)
    print(f"\n{title}:")
    print("=" * 50)
    for label, count in distribution.items():
        percentage = (count / total) * 100
        print(f"{label}: {count} ({percentage:.2f}%)")

def train_logistic_regression():
    try:
        # Charger les données
        raw_data = pd.DataFrame(list(load_data_from_mongodb()))
        print(f"Nombre d'éléments dans raw_data : {len(raw_data)}")
        print("\nAperçu des données brutes :")
        print(raw_data.head(10))

        # Vérifier les colonnes requises
        required_columns = ['cleaned_text', 'sentiment']
        missing_columns = [col for col in required_columns if col not in raw_data.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes : {', '.join(missing_columns)}")

        # Analyser la distribution initiale
        print("\nDistribution initiale des sentiments:")
        analyze_class_distribution(raw_data['sentiment'])

        # Préparation des données
        X_train, X_test, y_train, y_test, vectorizer = prepare_data_for_training(
            raw_data, "cleaned_text", "sentiment"
        )

        print(f"\nTaille de l'ensemble d'entraînement : {X_train.shape}")
        print(f"Taille de l'ensemble de test : {X_test.shape}")

        # Vérifier la distribution dans les ensembles d'entraînement et de test
        print("\nDistribution dans l'ensemble d'entraînement:")
        analyze_class_distribution(y_train, "Distribution - Training Set")
        
        print("\nDistribution dans l'ensemble de test:")
        analyze_class_distribution(y_test, "Distribution - Test Set")

        # Entraîner le modèle de régression logistique
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Évaluation du modèle
        accuracy = model.score(X_test, y_test)
        print(f"\nPrécision du modèle : {accuracy:.4f}")

        # Prédictions et rapport de classification
        y_pred = model.predict(X_test)
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred))

        # Afficher quelques exemples de prédictions
        print("\nComparaison des résultats sur quelques exemples de l'ensemble de test :")
        X_test_original = vectorizer.inverse_transform(X_test)
        for i in range(min(10, len(X_test_original))):
            print(f"\nTexte {i+1}:")
            print(f"  - Contenu : {' '.join(X_test_original[i])}")
            print(f"  - Sentiment réel : {y_test[i]}")
            print(f"  - Sentiment prédit : {y_pred[i]}")

        # Créer le répertoire si nécessaire et sauvegarder le modèle
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer_lr.pkl"))

        print("\nModèle de régression logistique entraîné et sauvegardé avec succès.")
        print(f"Chemin du modèle : {MODEL_PATH}")
        print(f"Chemin du vectorizer : {os.path.join(MODEL_DIR, 'vectorizer_lr.pkl')}")

        return model, vectorizer

    except Exception as e:
        print(f"\nErreur lors de l'entraînement : {str(e)}")
        raise

if __name__ == "__main__":
    train_logistic_regression()
