from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import os
import sys
import pandas as pd
import torch
import joblib
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import TextClassificationPipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_handling.data_loader import load_data_from_mongodb
from nlp.sentiment_analysis import analyze_sentiment

# Chemin pour sauvegarder le modèle
MODEL_DIR = os.path.join("..", "models/saved_models/bert")
MODEL_PATH = os.path.join(MODEL_DIR, "bert_model")

def balance_classes(data, target_column):
    classes = data[target_column].unique()
    class_counts = data[target_column].value_counts()
    min_count = class_counts.min()

    balanced_data = pd.DataFrame()

    for label in classes:
        class_subset = data[data[target_column] == label]
        balanced_subset = resample(class_subset, replace=True, n_samples=min_count, random_state=42)
        balanced_data = pd.concat([balanced_data, balanced_subset])

    return balanced_data.sample(frac=1, random_state=42)

def analyze_class_distribution(y, title="Distribution des classes"):
    distribution = pd.Series(y).value_counts()
    total = len(y)
    print(f"\n{title}:")
    print("=" * 50)
    for label, count in distribution.items():
        percentage = (count / total) * 100
        print(f"{label}: {count} ({percentage:.2f}%)")

def train_bert_classifier():
    try:
        raw_data = pd.DataFrame(list(load_data_from_mongodb()))
        print(f"Nombre d'éléments dans raw_data : {len(raw_data)}")

        required_columns = ['cleaned_text', 'sentiment']
        missing_columns = [col for col in required_columns if col not in raw_data.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes : {', '.join(missing_columns)}")

        print("\nDistribution initiale des sentiments:")
        analyze_class_distribution(raw_data['sentiment'])

        balanced_data = balance_classes(raw_data, 'sentiment')
        print("\nDistribution après équilibrage des classes:")
        analyze_class_distribution(balanced_data['sentiment'])

        X = balanced_data["cleaned_text"].tolist()
        y = balanced_data["sentiment"].tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

        train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
        test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

        train_labels = torch.tensor(y_train)
        test_labels = torch.tensor(y_test)

        training_args = TrainingArguments(
            output_dir=MODEL_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_dir=os.path.join(MODEL_DIR, 'logs'),
            evaluation_strategy="epoch",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_encodings,
            eval_dataset=test_encodings,
        )

        trainer.train()

        # Evaluation
        y_pred = trainer.predict(test_encodings)
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred.argmax(axis=-1)))

        # Sauvegarder le modèle et le tokenizer
        model.save_pretrained(MODEL_PATH)
        tokenizer.save_pretrained(MODEL_PATH)

        print("\nModèle BERT entraîné et sauvegardé avec succès.")
        print(f"Chemin du modèle : {MODEL_PATH}")

    except Exception as e:
        print(f"\nErreur lors de l'entraînement : {str(e)}")
        raise

if __name__ == "__main__":
    train_bert_classifier()
