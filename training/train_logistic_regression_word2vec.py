import os
import sys
import random
import numpy as np
import pandas as pd
import joblib

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_handling.data_loader import load_data_from_mongodb

def train_word2vec(sentences, vector_size=200, window=7, min_count=2, epochs=20):
    """Entraîne le modèle Word2Vec avec des paramètres optimisés."""
    print("Entraînement de Word2Vec...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1,  # Skip-gram
        epochs=epochs,
    )
    return model

def vectorize_sentences(sentences, w2v_model):
    """Vectorise les phrases en utilisant Word2Vec avec gestion des mots hors vocabulaire."""
    print("Vectorisation des phrases...")
    vectorized = []

    for sentence in sentences:
        words = sentence.split()
        word_vectors = []

        for word in words:
            if word in w2v_model.wv:
                word_vectors.append(w2v_model.wv[word])

        if word_vectors:
            sentence_vector = np.mean(word_vectors, axis=0)
        else:
            sentence_vector = np.zeros(w2v_model.vector_size)
        
        vectorized.append(sentence_vector)

    return np.array(vectorized)

def balance_and_augment_data(data, text_column, label_column):
    """Équilibre et augmente les données pour les classes minoritaires."""
    print("Équilibrage des données...")
    ros = RandomOverSampler(random_state=42)

    X = data[text_column]
    y = data[label_column]

    # Vérification de la distribution initiale
    class_counts = pd.Series(y).value_counts()
    print(f"Distribution initiale : {class_counts.to_dict()}")

    # Équilibrage avec RandomOverSampler
    X_resampled, y_resampled = ros.fit_resample(pd.DataFrame(X), pd.Series(y))

    balanced_data = pd.DataFrame({
        text_column: X_resampled.iloc[:, 0],
        label_column: y_resampled
    })

    # Vérification de la distribution après équilibrage
    class_counts_balanced = balanced_data[label_column].value_counts()
    print(f"Distribution après équilibrage : {class_counts_balanced.to_dict()}")

    return balanced_data

def train_model(balanced_data, text_column, label_column, base_save_dir):
    """Pipeline complet d'entraînement avec sauvegarde des modèles."""
    # Préparation des données
    sentences = balanced_data[text_column].tolist()
    labels = balanced_data[label_column].tolist()

    # Split des données
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        sentences, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Tokenisation pour Word2Vec
    tokenized_sentences = [text.split() for text in X_train_texts]

    # Entraînement Word2Vec
    w2v_model = train_word2vec(tokenized_sentences)

    # Vectorisation des données
    X_train = vectorize_sentences(X_train_texts, w2v_model)
    X_test = vectorize_sentences(X_test_texts, w2v_model)

    # Scaling des caractéristiques
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entraînement du classificateur
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=5000,
        multi_class="multinomial",
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    )
    lr_model.fit(X_train_scaled, y_train)

    # Évaluation du modèle
    y_pred = lr_model.predict(X_test_scaled)
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    # Sauvegarde des modèles
    model_dir = os.path.join(base_save_dir, "word2vec")
    os.makedirs(model_dir, exist_ok=True)

    vectorizer_path = os.path.join(model_dir, "vectorizer_word2vec.model")
    model_path = os.path.join(model_dir, "word2vec.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    w2v_model.save(vectorizer_path)
    joblib.dump(lr_model, model_path, compress=0)
    joblib.dump(scaler, scaler_path)

    print(f"Modèles sauvegardés dans {model_dir}")

    return w2v_model, lr_model, scaler

def main():
    # Répertoire de sauvegarde
    BASE_SAVE_DIR = "/Users/issa/Desktop/moodlyze/models/models/saved_models"
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    # Chargement des données
    print("Chargement des données...")
    data = pd.DataFrame(list(load_data_from_mongodb()))

    # Équilibrage et augmentation des données
    balanced_data = balance_and_augment_data(
        data, 
        text_column="cleaned_text", 
        label_column="sentiment"
    )

    # Entraînement du modèle
    train_model(
        balanced_data,
        text_column="cleaned_text",
        label_column="sentiment", 
        base_save_dir=BASE_SAVE_DIR
    )

if __name__ == "__main__":
    main()
