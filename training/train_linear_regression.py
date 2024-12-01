import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_handling.data_loader import load_data_from_mongodb
from pandas import DataFrame
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def train_linear_regression():
    df = pd.DataFrame(load_data_from_mongodb())
    
    vectorizer = CountVectorizer()
    vectorized_texts = []
    
    for text in df['cleaned_text']:
        # Vectorise chaque texte individuellement
        vectorized_text = vectorizer.fit_transform([text])
        vectorized_texts.append(vectorized_text)
    
    # Optionnel : afficher quelques informations
    print("Nombre de textes vectorisés :", len(vectorized_texts))
    print("Exemple de premier texte vectorisé :", vectorized_texts[0].toarray())
    
    # Récupérer le vocabulaire
    vocabulary = vectorizer.get_feature_names_out()
    print("Vocabulaire :", vocabulary)

if __name__ == "__main__":
    train_linear_regression()