from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_handling.data_loader import load_data_from_mongodb
from data_handling.data_preprocessor import prepare_data_for_training

MODEL_DIR = os.path.join("..", "models/saved_models/logistic_regression")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")

def normalize_sentiment_labels(data, sentiment_column='sentiment'):
    """
    Normalise les étiquettes de sentiment pour assurer qu'on a bien 3 classes.
    """
    # Mapping des différentes formes possibles vers les 3 classes standard
    sentiment_mapping = {
        'positive': 'positive',
        'positif': 'positive',
        'pos': 'positive',
        'negative': 'negative',
        'negatif': 'negative',
        'neg': 'negative',
        'neutral': 'neutral',
        'neutre': 'neutral',
        'neu': 'neutral'
    }
    
    # Convertir en minuscules et normaliser
    data[sentiment_column] = data[sentiment_column].str.lower()
    data[sentiment_column] = data[sentiment_column].map(sentiment_mapping)
    
    # Vérifier les valeurs uniques après normalisation
    unique_sentiments = data[sentiment_column].unique()
    print("\nSentiments uniques après normalisation:", unique_sentiments)
    
    # Vérifier qu'on a bien les 3 classes attendues
    expected_sentiments = {'positive', 'negative', 'neutral'}
    missing_sentiments = expected_sentiments - set(unique_sentiments)
    if missing_sentiments:
        print(f"\nAttention: Classes de sentiment manquantes: {missing_sentiments}")
    
    unexpected_sentiments = set(unique_sentiments) - expected_sentiments
    if unexpected_sentiments:
        print(f"\nAttention: Classes de sentiment inattendues: {unexpected_sentiments}")
        # Optionnel: filtrer les sentiments inattendus
        data = data[data[sentiment_column].isin(expected_sentiments)]
    
    return data

def ensure_directory_exists(directory):
    """Crée le répertoire s'il n'existe pas."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def analyze_sentiment_distribution(data, sentiment_column='sentiment'):

    all_sentiments = ['positive', 'neutral', 'negative']

    # Calculer la distribution des sentiments
    sentiment_counts = data[sentiment_column].value_counts()

    for sentiment in all_sentiments:
        if sentiment not in sentiment_counts.index:
            sentiment_counts[sentiment] = 0

    sentiment_counts = sentiment_counts.reindex(all_sentiments)

    sentiment_percentages = (sentiment_counts / len(data) * 100).round(2)
    
    # Créer un DataFrame avec les statistiques
    distribution_stats = pd.DataFrame({
        'Count': sentiment_counts,
        'Percentage': sentiment_percentages
    })
    
    # Arrondir les pourcentages à 2 décimales
    distribution_stats['Percentage'] = distribution_stats['Percentage'].round(2)
    
    # Afficher les statistiques
    print("\nDistribution des sentiments:")
    print("=" * 50)
    print(distribution_stats)
    print("\nNombre total d'échantillons:", len(data))
    
    # Créer un graphique de la distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=distribution_stats.index, y='Count', data=distribution_stats)
    plt.title('Distribution des Sentiments')
    plt.xlabel('Sentiment')
    plt.ylabel('Nombre d\'échantillons')
    
    # Ajouter les pourcentages au-dessus des barres
    for i, v in enumerate(distribution_stats['Count']):
        plt.text(i, v, f'{distribution_stats["Percentage"][i]}%', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return distribution_stats

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
        
        print("\nDistribution des sentiments avant prétraitement:")
        initial_distribution = analyze_sentiment_distribution(raw_data)
    

        # Préparation des données
        X_train, X_test, y_train, y_test, vectorizer = prepare_data_for_training(
            raw_data, "cleaned_text", "sentiment"
        )

        print(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]}")
        print(f"Taille de l'ensemble de test : {X_test.shape[0]}")

        # Configuration de la recherche par grille
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'class_weight': ['balanced', None],
            'max_iter': [1000, 2000]
        }

        # Entraînement du modèle
        base_model = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(base_model, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        # Évaluation du modèle
        model = grid_search.best_estimator_
        accuracy = model.score(X_test, y_test)
        print(f"\nMeilleurs paramètres : {grid_search.best_params_}")
        print(f"Précision du modèle : {accuracy:.4f}")

        # Prédictions et rapport de classification
        y_pred = model.predict(X_test)
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred))

        # Affichage des exemples
        X_test_original = vectorizer.inverse_transform(X_test)
        print("\nComparaison des résultats sur quelques exemples de l'ensemble de test :")
        for i in range(min(10, len(X_test_original))):
            print(f"\nTweet {i+1}:")
            print(f"  - Texte : {' '.join(X_test_original[i])}")
            print(f"  - Sentiment réel : {y_test[i]}")
            print(f"  - Sentiment prédit : {y_pred[i]}")

        # Sauvegarde du modèle et du vectoriseur
        ensure_directory_exists(MODEL_DIR)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer_lr.pkl"))

        print("\nModèle de régression logistique entraîné et sauvegardé avec succès.")

        print("\nDistribution des sentiments dans l'ensemble d'entraînement:")
        train_data = pd.DataFrame({'sentiment': y_train})
        train_distribution = analyze_sentiment_distribution(train_data)
        
        print("\nDistribution des sentiments dans l'ensemble de test:")
        test_data = pd.DataFrame({'sentiment': y_test})
        test_distribution = analyze_sentiment_distribution(test_data)
    
        
        # Retourner le modèle et le vectoriseur pour utilisation ultérieure si nécessaire
        return model, vectorizer, {
        'initial_distribution': initial_distribution,
        'train_distribution': train_distribution,
        'test_distribution': test_distribution
    }

    except Exception as e:
        print(f"\nErreur lors de l'entraînement : {str(e)}")
        raise

if __name__ == "__main__":
    train_logistic_regression()