from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nlp.sentiment_analysis import analyze_sentiment

def prepare_data_for_training(df, text_column='cleaned_text', target_column='label'):
    df[target_column] = df[text_column].apply(analyze_sentiment)
    df = df.dropna(subset=[target_column])
    df.head(10)


    vectorizer = CountVectorizer(analyzer=lambda x: x)
    X = vectorizer.fit_transform(df[text_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, vectorizer
