import logging
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

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chemins pour sauvegarder le modèle et le vectorizer
MODEL_DIR = os.path.join("..", "models", "saved_models", "word2vec")
MODEL_PATH = os.path.join(MODEL_DIR, "word2vec.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer_word2vec.pkl")

def balance_classes(data, target_column):
    """Équilibre les classes du DataFrame."""
    classes = data[target_column].unique()
    min_count = data[target_column].value_counts().min()

    balanced_data = pd.DataFrame()
    for label in classes:
        class_subset = data[data[target_column] == label]
        balanced_subset = resample(class_subset, replace=True, n_samples=min_count, random_state=42)
        balanced_data = pd.concat([balanced_data, balanced_subset])

    return balanced_data.sample(frac=1, random_state=42)

def analyze_class_distribution(y, title="Distribution des classes"):
    """Affiche la distribution des classes dans le dataset."""
    distribution = pd.Series(y).value_counts()
    total = len(y)
    logging.info(f"\n{title}:")
    logging.info("=" * 50)
    for label, count in distribution.items():
        percentage = (count / total) * 100
        logging.info(f"{label}: {count} ({percentage:.2f}%)")

def save_model_and_vectorizer(model, vectorizer):
    """Sauvegarde le modèle et le vectorizer."""
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Sauvegarder le modèle
        logging.info(f"Sauvegarde du modèle dans : {MODEL_PATH}")
        joblib.dump(model, MODEL_PATH)

        # Sauvegarder le vectorizer
        logging.info(f"Sauvegarde du vectorizer dans : {VECTORIZER_PATH}")
        joblib.dump(vectorizer, VECTORIZER_PATH)

        logging.info("Sauvegarde du modèle et du vectorizer terminée avec succès.")

    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde : {str(e)}")
        raise

def train_logistic_regression():
    try:
        # Charger les données
        raw_data = pd.DataFrame(list(load_data_from_mongodb()))
        logging.info(f"Nombre d'éléments dans raw_data : {len(raw_data)}")

        # Vérifier les colonnes requises
        required_columns = ['cleaned_text', 'sentiment']
        missing_columns = [col for col in required_columns if col not in raw_data.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes : {', '.join(missing_columns)}")

        # Analyser la distribution initiale
        analyze_class_distribution(raw_data['sentiment'], "Distribution initiale des sentiments")

        # Équilibrer les classes
        balanced_data = balance_classes(raw_data, 'sentiment')
        analyze_class_distribution(balanced_data['sentiment'], "Distribution après équilibrage")

        X = balanced_data["cleaned_text"]
        y = balanced_data["sentiment"]

        # Séparer en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Vectorisation
        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')
        X_train_vect = vectorizer.fit_transform(X_train)
        X_test_vect = vectorizer.transform(X_test)

        # Entraîner le modèle
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vect, y_train)

        # Évaluer le modèle
        accuracy = model.score(X_test_vect, y_test)
        logging.info(f"Précision du modèle : {accuracy:.4f}")

        # Afficher le rapport de classification
        y_pred = model.predict(X_test_vect)
        logging.info("\nRapport de classification :")
        logging.info(classification_report(y_test, y_pred))

        # Sauvegarder le modèle et le vectorizer
        save_model_and_vectorizer(model, vectorizer)

        logging.info("Modèle de régression logistique entraîné et sauvegardé avec succès.")
        return model, vectorizer

    except Exception as e:
        logging.error(f"Erreur lors de l'entraînement : {str(e)}")
        raise

if __name__ == "__main__":
    train_logistic_regression()
