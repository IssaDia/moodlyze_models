import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import joblib
import os
import sys
from collections import Counter
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_handling.data_loader import load_data_from_mongodb

# Répertoire de sauvegarde des modèles
BASE_SAVE_DIR = "/Users/issa/Desktop/moodlyze/models/models/saved_models/word2vec"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)


def train_word2vec(sentences, vector_size=200, window=7, min_count=2, epochs=20):
    """Entraîne le modèle Word2Vec avec des paramètres optimisés."""
    print("Entraînement de Word2Vec...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1,
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
        word_counts = Counter(words)

        for word in words:
            if word in w2v_model.wv:
                tf = word_counts[word] / len(words)
                word_vectors.append(w2v_model.wv[word] * tf)

        if word_vectors:
            sentence_vector = np.mean(word_vectors, axis=0)
        else:
            sentence_vector = np.zeros(w2v_model.vector_size)
        vectorized.append(sentence_vector)

    return np.array(vectorized)


def train_logistic_regression(X, y):
    """Entraîne le modèle de régression logistique avec des paramètres optimisés."""
    print("Entraînement du modèle Logistic Regression...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        C=0.1,
        max_iter=5000,
        multi_class="multinomial",
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_scaled, y)
    return model, scaler


def augment_sentence(sentence, num_augmented=1):
    """Augmente une phrase en réorganisant les mots ou en ajoutant du bruit."""
    augmented_sentences = []
    for _ in range(num_augmented):
        words = sentence.split()
        random.shuffle(words)  # Mélange des mots
        augmented_sentences.append(" ".join(words))
    return augmented_sentences


def balance_and_augment_data(data, text_column, label_column, min_samples=1000):
    """Équilibre et augmente les données pour améliorer les performances."""
    print("Équilibrage et augmentation des données...")
    ros = RandomOverSampler(random_state=42)
    X = data[text_column].tolist()
    y = data[label_column].tolist()

    # Vérification de la taille du dataset
    class_counts = pd.Series(y).value_counts()
    print(f"Nombre d'exemples par classe avant l'augmentation: {class_counts.to_dict()}")
    
    if len(data) < min_samples:
        print(f"Dataset trop petit (moins de {min_samples} échantillons). Augmentation nécessaire.")
    else:
        print(f"Dataset suffisant (plus de {min_samples} échantillons).")

    # Rééquilibrage des classes
    X_resampled, y_resampled = ros.fit_resample(pd.DataFrame(X), pd.Series(y))
    
    # Augmentation des phrases
    augmented_sentences = X_resampled.iloc[:, 0].apply(lambda sentence: augment_sentence(sentence, num_augmented=1)).explode().tolist()

    # Création des données équilibrées et augmentées
    balanced_data = pd.DataFrame(
        {
            text_column: X_resampled.iloc[:, 0].tolist() + augmented_sentences,
            label_column: y_resampled.tolist() + y_resampled.tolist(),
        }
    )
    
    return balanced_data


def train_and_save_model(data, text_column, label_column, base_dir):
    """Pipeline complet d'entraînement avec des améliorations."""
    print("Préparation des données...")
    sentences = data[text_column].tolist()
    labels = data[label_column].tolist()

    word2vec_dir = os.path.join(base_dir, "word2vec")
    os.makedirs(word2vec_dir, exist_ok=True)

    vectorizer_path = os.path.join(word2vec_dir, "vectorizer_word2vec.pkl")
    model_path = os.path.join(word2vec_dir, "word2vec.pkl")
    scaler_path = os.path.join(word2vec_dir, "scaler.pkl")

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        sentences, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Entraînement du modèle Word2Vec...")
    tokenized_sentences = [text.split() for text in X_train_texts]
    w2v_model = train_word2vec(tokenized_sentences)

    print("Vectorisation des données...")
    X_train = vectorize_sentences(X_train_texts, w2v_model)
    X_test = vectorize_sentences(X_test_texts, w2v_model)

    lr_model, scaler = train_logistic_regression(X_train, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = lr_model.predict(X_test_scaled)
    print("\nÉvaluation du modèle:")
    print(classification_report(y_test, y_pred))

    print(f"Sauvegarde des modèles dans {word2vec_dir}...")
    joblib.dump(w2v_model, vectorizer_path)
    joblib.dump(lr_model, model_path)
    joblib.dump(scaler, scaler_path)

    print("Modèles sauvegardés avec succès!")
    return w2v_model, lr_model, scaler


if __name__ == "__main__":
    print("Chargement des données depuis MongoDB...")
    data = pd.DataFrame(list(load_data_from_mongodb()))

    balanced_data = balance_and_augment_data(
        data, text_column="cleaned_text", label_column="sentiment"
    )

    base_dir = "/Users/issa/Desktop/moodlyze/models/models/saved_models"

    w2v_model, lr_model, scaler = train_and_save_model(
        balanced_data,
        text_column="cleaned_text",
        label_column="sentiment",
        base_dir=base_dir,
    )
