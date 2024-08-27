import joblib

def save_model(model, vectorizer, model_path='saved_model.pkl', vectorizer_path='vectorizer.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model and vectorizer saved to {model_path} and {vectorizer_path}")
