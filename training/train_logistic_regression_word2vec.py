import numpy as np
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_handling.data_loader import load_data_from_mongodb

BASE_SAVE_DIR = "/Users/issa/Desktop/moodlyze/models/models/saved_models/word2vec"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)


# Charger ou entraîner un modèle Word2Vec
def train_word2vec(sentences, vector_size=100, window=5, min_count=1):
    print("Entraînement de Word2Vec...")
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return model

# Moyenne des vecteurs Word2Vec pour chaque phrase
def vectorize_sentences(sentences, w2v_model):
    print("Vectorisation des phrases...")
    vectorized = []
    for sentence in sentences:
        words = sentence.split()
        word_vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if len(word_vectors) > 0:
            vectorized.append(np.mean(word_vectors, axis=0))
        else:
            # Si aucun mot n'a de vecteur, on retourne un vecteur nul
            vectorized.append(np.zeros(w2v_model.vector_size))
    return np.array(vectorized)

# Entraîner un modèle Logistic Regression
def train_logistic_regression(X, y):
    print("Entraînement du modèle Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

# Pipeline complet pour Word2Vec + Logistic Regression
def train_and_save_model(data, text_column, label_column, w2v_path, model_path, vectorizer_path):
    # Préparation des données
    print("Chargement des données...")
    sentences = data[text_column].tolist()
    labels = data[label_column].tolist()
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        sentences, labels, test_size=0.1, random_state=42, stratify=labels
    )

    # Entraîner ou charger un modèle Word2Vec
    if os.path.exists(w2v_path):
        print(f"Chargement du modèle Word2Vec depuis {w2v_path}...")
        w2v_model = KeyedVectors.load(w2v_path)
    else:
        w2v_model = train_word2vec([sentence.split() for sentence in X_train_texts])
        w2v_model.save(w2v_path)

    # Vectoriser les phrases
    X_train = vectorize_sentences(X_train_texts, w2v_model)
    X_test = vectorize_sentences(X_test_texts, w2v_model)

    # Entraîner le modèle Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)

    # Évaluer le modèle
    print("Évaluation du modèle...")
    y_pred = lr_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Sauvegarder le modèle et le vectorizer
    print(f"Sauvegarde du modèle dans {model_path}...")
    joblib.dump(lr_model, model_path)

    print(f"Sauvegarde du Word2Vec dans {vectorizer_path}...")
    joblib.dump(w2v_model, vectorizer_path)

    print("Modèle et vectorizer sauvegardés avec succès !")

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger les données
    data = pd.DataFrame(list(load_data_from_mongodb()))

    # Chemins pour sauvegarder le modèle et le vectorizer
    w2v_path = os.path.join(BASE_SAVE_DIR, "word2vec.kv")
    model_path = os.path.join(BASE_SAVE_DIR, "word2vec.pkl")
    vectorizer_path = os.path.join(BASE_SAVE_DIR, "word2vec_vectorizer.pkl")

    # Entraîner et sauvegarder le modèle
    train_and_save_model(
        data,
        text_column="cleaned_text",
        label_column="sentiment",
        w2v_path=w2v_path,
        model_path=model_path,
        vectorizer_path=vectorizer_path
    )
