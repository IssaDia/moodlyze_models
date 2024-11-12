from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import os
import sys
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_handling.data_loader import load_data_from_mongodb
from nlp.sentiment_analysis import analyze_sentiment

# Path to save the trained model
MODEL_DIR = os.path.join("..", "models/saved_models/naive_bayes")
MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes.pkl")

def balance_classes(data, target_column):
    # Diviser le dataset par classe
    classes = data[target_column].unique()
    class_counts = data[target_column].value_counts()
    min_count = class_counts.min()  # Trouver le nombre d'éléments de la classe minoritaire

    balanced_data = pd.DataFrame()  # DataFrame pour stocker les échantillons équilibrés

    for label in classes:
        class_subset = data[data[target_column] == label]
        balanced_subset = resample(class_subset, replace=True, n_samples=min_count, random_state=42)  # changez replace=False à True pour un sur-échantillonnage
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

def train_naive_bayes():
    try:
        # Charger et convertir les données en DataFrame
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

        # Appliquer l'analyse de sentiment
        try:
            raw_data["cleaned_text"] = raw_data["sentiment"].apply(analyze_sentiment)
        except Exception as e:
            print(f"Erreur lors de l'analyse de sentiment : {e}")
            return None  # Ou gérer autrement l'erreur

        # Supprimer les lignes où l'analyse de sentiment a échoué (si applicable)
        raw_data = raw_data[raw_data["cleaned_text"].notnull() & (raw_data["cleaned_text"] != '')]
        print(f"Nombre de documents après analyse de sentiment : {len(raw_data)}")

        # Vérifier s'il reste suffisamment de données
        if len(raw_data) < 100:
            print("Attention : Le nombre de documents restants est très faible.")

        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')

        X_train = vectorizer.fit_transform(X_train)  
        X_test = vectorizer.transform(X_test)  

        # Entraîner le modèle Naive Bayes
        model = MultinomialNB(alpha=1.0) 
        model.fit(X_train, y_train)

        # Évaluation du modèle
        accuracy = model.score(X_test, y_test)
        print(f"\nPrécision du modèle : {accuracy:.4f}")

        # Prédictions et rapport de classification
        y_pred = model.predict(X_test)
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred))

        print(f"Valeurs prédites uniques : {np.unique(y_pred)}")

        # Créer le répertoire si nécessaire et sauvegarder le modèle
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer_nb.pkl"))

        print("\nModèle Naive Bayes entraîné et sauvegardé avec succès.")
        print(f"Chemin du modèle : {MODEL_PATH}")
        print(f"Chemin du vectorizer : {os.path.join(MODEL_DIR, 'vectorizer_nb.pkl')}")

        return model, vectorizer

    except Exception as e:
        print(f"\nErreur lors de l'entraînement : {str(e)}")
        raise

if __name__ == "__main__":
    train_naive_bayes()
