from sklearn.svm import SVC
import os
import sys

# Ajouter le chemin vers le dossier parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
from data_handling.data_loader import load_data_from_mongodb
from data_handling.data_preprocessor import prepare_data_for_training

# Chemin pour sauvegarder le modèle entraîné
MODEL_DIR = os.path.join("..", "savedModels")
MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")

def train_svm():
    raw_data = load_data_from_mongodb() 
    X_train, X_test, y_train, y_test, vectorizer = prepare_data_for_training(raw_data) 

    # Entraîner le modèle SVM
    model = SVC(kernel='linear', max_iter=1000)  # Kernel linéaire pour commencer
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Précision du modèle SVM : {accuracy:.4f}")

    y_pred = model.predict(X_test)
    X_test_original = vectorizer.inverse_transform(X_test.toarray())

    print("\nComparaison des résultats sur quelques exemples de l'ensemble de test :")
    for i in range(10):
        print(f"Tweet {i+1}:")
        print(f"  - Texte : {' '.join(X_test_original[i])}")  
        print(f"  - Sentiment réel : {y_test.iloc[i]}")
        print(f"  - Sentiment prédit : {y_pred[i]}")
        print()

    # Sauvegarder le modèle et le vectoriseur
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, os.path.join("../savedModels", "vectorizer_svm.pkl"))

    print("Modèle SVM entraîné et sauvegardé avec succès.")

if __name__ == "__main__":
    train_svm()
