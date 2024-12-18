from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0:
        return 2  # Positif
    elif polarity < 0:
        return 0  # Négatif
    else:
        return 1  # Neutre