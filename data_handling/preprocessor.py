from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nlp.sentiment_analysis import analyze_sentiment

def analyzer_function(tokens):
    return tokens

def prepare_data_for_training(df, text_column='tokens', target_column='label'):
    df['text_for_sentiment'] = df[text_column].apply(lambda tokens: ' '.join(tokens))
    df[target_column] = df['text_for_sentiment'].apply(analyze_sentiment)
    df = df.dropna(subset=[target_column])

    # Utiliser CountVectorizer avec les tokens
    vectorizer = CountVectorizer(analyzer=analyzer_function)    
    X = vectorizer.fit_transform(df[text_column])
    y = df[target_column]

    # Séparation des données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, vectorizer
