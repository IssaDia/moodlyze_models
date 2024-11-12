from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_handling.data_loader import load_data_from_mongodb

# Chemin pour sauvegarder le modèle
MODEL_DIR = os.path.join("..", "models/saved_models/logistic_regression")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")

def balance_classes(data, target_column):
    # Diviser le dataset par classe
    classes = data[target_column].unique()
    class_counts = data[target_column].value_counts()
    min_count = class_counts.min()  # Trouver le nombre d'éléments de la classe minoritaire

    balanced_data = pd.DataFrame()  # DataFrame pour stocker les échantillons équilibrés

    for label in classes:
        class_subset = data[data[target_column] == label]
        balanced_subset = resample(class_subset, replace=True, n_samples=min_count, random_state=42)
        balanced_data = pd.concat([balanced_data, balanced_subset])

    return balanced_data.sample(frac=1, random_state=42) 

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

        # Vérifier les colonnes requises
        required_columns = ['cleaned_text', 'sentiment']
        missing_columns = [col for col in required_columns if col not in raw_data.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes : {', '.join(missing_columns)}")

        # Analyser la distribution initiale
        print("\nDistribution initiale des sentiments:")
        analyze_class_distribution(raw_data['sentiment'])

        # Équilibrer les classes
        balanced_data = balance_classes(raw_data, 'sentiment')
        print("\nDistribution après équilibrage des classes:")
        analyze_class_distribution(balanced_data['sentiment'])

        X = balanced_data["cleaned_text"]
        y = balanced_data["sentiment"]

        # Séparer en ensembles d'entraînement et de test avec stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nTaille de l'ensemble d'entraînement : {X_train.shape}")
        print(f"Taille de l'ensemble de test : {X_test.shape}")

        # Vérifier la distribution dans les ensembles d'entraînement et de test
        print("\nDistribution dans l'ensemble d'entraînement:")
        analyze_class_distribution(y_train, "Distribution - Training Set")
        
        print("\nDistribution dans l'ensemble de test:")
        analyze_class_distribution(y_test, "Distribution - Test Set")

        # Vectorisation
        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

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
