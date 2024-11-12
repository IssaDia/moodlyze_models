from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nlp.sentiment_analysis import analyze_sentiment
import pandas as pd

def debug_data(df, text_column='cleaned_text'):
    """Fonction de débogage pour afficher des informations sur la colonne spécifiée."""
    print(f"\nDébogage de la colonne {text_column}:")
    print(f"Nombre total d'éléments : {len(df)}")
    print(f"Nombre de valeurs non-null : {df[text_column].count()}")
    print(f"Nombre de valeurs null : {df[text_column].isnull().sum()}")
    print(f"Nombre de chaînes vides : {(df[text_column] == '').sum()}")
    print(f"Nombre de chaînes contenant uniquement des espaces : {(df[text_column].str.isspace()).sum()}")

def prepare_data_for_training(data, text_column='cleaned_text', target_column='label', test_size=0.2):
    if isinstance(data, pd.Series):
        raise ValueError("Les données passées doivent être un DataFrame, pas une série.")
    
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data

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
    try:
        df[target_column] = df[text_column].apply(analyze_sentiment)
    except Exception as e:
        print(f"Erreur lors de l'analyse de sentiment : {e}")
        return None  # Ou gérer autrement l'erreur

    # Supprimer les lignes où l'analyse de sentiment a échoué (si applicable)
    df = df[df[target_column].notnull() & (df[target_column] != '')]
    print(f"Nombre de documents après analyse de sentiment : {len(df)}")
    
    # Vérifier s'il reste suffisamment de données
    if len(df) < 100:
        print("Attention : Le nombre de documents restants est très faible.")
    
    # Utiliser TfidfVectorizer
    vectorizer = TfidfVectorizer()    
    X = vectorizer.fit_transform(df[text_column]) 
    y = df[target_column]
    
    # Encodage des cibles si nécessaire
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Séparation des données en ensemble d'entraînement et de test avec échantillonnage stratifié
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    print(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]}")
    print(f"Taille de l'ensemble de test : {X_test.shape[0]}")
    
    # Afficher la distribution des classes
    print("Distribution - Training Set:")
    print(pd.Series(y_train).value_counts(normalize=True) * 100)
    print("\nDistribution - Test Set:")
    print(pd.Series(y_test).value_counts(normalize=True) * 100)
    
    return X_train, X_test, y_train, y_test, vectorizer
