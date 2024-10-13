from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nlp.sentiment_analysis import analyze_sentiment

def analyzer_function(tokens):
    return tokens

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nlp.sentiment_analysis import analyze_sentiment
import pandas as pd

def debug_data(df, text_column='cleaned_text', start_row=200, end_row=210):
    print(f"\nDébogage de la colonne {text_column}:")
    print(f"Nombre total d'éléments : {len(df)}")
    print(f"Nombre de valeurs non-null : {df[text_column].count()}")
    print(f"Nombre de valeurs null : {df[text_column].isnull().sum()}")
    print(f"Nombre de chaînes vides : {(df[text_column] == '').sum()}")
    print(f"Nombre de chaînes contenant uniquement des espaces : {(df[text_column].str.isspace()).sum()}")
    
    print(f"\nAffichage des lignes {start_row} à {end_row}:")
    print(df.iloc[start_row:end_row+1])
    
    print("\nValeurs de la colonne 'cleaned_text' pour ces lignes:")
    for idx, row in df.iloc[start_row:end_row+1].iterrows():
        print(f"Index {idx}, cleaned_text: '{row[text_column]}'")

def prepare_data_for_training(data, text_column='cleaned_text', target_column='label'):
    # Convertir en DataFrame si ce n'est pas déjà le cas
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    
    print(f"Nombre initial de documents : {len(df)}")
    
    # Débogage initial
    debug_data(df, text_column)
    
    # Remplacer les valeurs NaN par une chaîne vide
    df[text_column] = df[text_column].fillna('')
    
    # Supprimer les lignes où le texte est vide ou contient uniquement des espaces
    df = df[~(df[text_column] == '') & ~(df[text_column].str.isspace())]
    print(f"Nombre de documents après nettoyage de {text_column}: {len(df)}")
    
    # Débogage après nettoyage
    debug_data(df, text_column)
    
    # Appliquer l'analyse de sentiment
    df[target_column] = df[text_column].apply(analyze_sentiment)
    
    # Supprimer les lignes où l'analyse de sentiment a échoué (si applicable)
    df = df.dropna(subset=[target_column])
    print(f"Nombre de documents après analyse de sentiment : {len(df)}")
    
    # Vérifier s'il reste suffisamment de données
    if len(df) < 100:
        print("Attention : Le nombre de documents restants est très faible.")
    
    # Utiliser CountVectorizer avec le texte complet
    vectorizer = CountVectorizer()    
    X = vectorizer.fit_transform(df[text_column]) 
    y = df[target_column]
    
    # Séparation des données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]}")
    print(f"Taille de l'ensemble de test : {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, vectorizer


