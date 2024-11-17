from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import os
import sys
import joblib
import pandas as pd
import numpy as np

nltk.download('punkt', quiet=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_handling.data_loader import load_data_from_mongodb

MODEL_DIR = os.path.join("..", "models/saved_models/word2vec")
MODEL_PATH = os.path.join(MODEL_DIR, "word2vec.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer_lr.pkl")

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """Vectorizer compatible avec scikit-learn utilisant Word2Vec"""
    
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.w2v_model = None
        
    def fit(self, X, y=None):
        # Tokenisation et entraînement du modèle Word2Vec
        tokenized_texts = [word_tokenize(str(text).lower()) for text in X]
        self.w2v_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4
        )
        return self
        
    def transform(self, X):
        # Conversion des textes en vecteurs
        vectors = []
        for text in X:
            tokens = word_tokenize(str(text).lower())
            word_vectors = [self.w2v_model.wv[word] 
                          for word in tokens 
                          if word in self.w2v_model.wv]
            
            if word_vectors:
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                doc_vector = np.zeros(self.vector_size)
            
            vectors.append(doc_vector)
        
        return np.array(vectors)

def balance_classes(data, target_column):
    classes = data[target_column].unique()
    class_counts = data[target_column].value_counts()
    min_count = class_counts.min()

    balanced_data = pd.DataFrame()

    for label in classes:
        class_subset = data[data[target_column] == label]
        balanced_subset = resample(class_subset, replace=True, n_samples=min_count, random_state=42)
        balanced_data = pd.concat([balanced_data, balanced_subset])

    return balanced_data.sample(frac=1, random_state=42)

def analyze_class_distribution(y, title="Distribution des classes"):
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

        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nTaille de l'ensemble d'entraînement : {len(X_train)}")
        print(f"Taille de l'ensemble de test : {len(X_test)}")

        # Création et entraînement du vectorizer Word2Vec
        print("\nCréation et entraînement du vectorizer Word2Vec...")
        vectorizer = Word2VecVectorizer(vector_size=100, window=5, min_count=1)
        
        # Transformation des données
        print("Transformation des données...")
        X_train_vectors = vectorizer.fit_transform(X_train)
        X_test_vectors = vectorizer.transform(X_test)

        # Entraînement de la régression logistique
        print("Entraînement du modèle de régression logistique...")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vectors, y_train)

        # Évaluation du modèle
        accuracy = model.score(X_test_vectors, y_test)
        print(f"\nPrécision du modèle : {accuracy:.4f}")

        # Prédictions et rapport de classification
        y_pred = model.predict(X_test_vectors)
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred))

        # Sauvegarder les modèles
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)

        print("\nModèles entraînés et sauvegardés avec succès.")
        print(f"Chemin du modèle : {MODEL_PATH}")
        print(f"Chemin du vectorizer : {VECTORIZER_PATH}")

        return model, vectorizer

    except Exception as e:
        print(f"\nErreur lors de l'entraînement : {str(e)}")
        raise

def predict_sentiment(text, vectorizer, model):
    """Prédit le sentiment d'un nouveau texte."""
    vector = vectorizer.transform([text])
    return model.predict(vector)[0]

if __name__ == "__main__":
    # Entraîner les modèles
    lr_model, vectorizer = train_logistic_regression()
    
    # Test avec quelques exemples
    test_texts = [
        "This is absolutely fantastic!",
        "I'm really disappointed with this service",
        "It's okay, nothing special",
        "Thanks for wasting my time, you're so helpful! (sarcastique)",
        "Another wonderful day at work... (ironique)"
    ]
    
    print("\nTests de prédiction :")
    print("=" * 50)
    for text in test_texts:
        sentiment = predict_sentiment(text, vectorizer, lr_model)
        print(f"\nTexte: {text}")
        print(f"Sentiment prédit: {sentiment}")