from textblob import TextBlob

def analyze_sentiment(text):
    """Analyse le sentiment d'un texte avec TextBlob et retourne 1 pour positif, 0 pour négatif."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 1  # Positif
    elif polarity < 0:
        return 0  # Négatif
    else:
        return None