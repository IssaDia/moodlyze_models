import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
import sys

# Désactiver les GPU et MPS
os.environ["PYTORCH_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


torch.set_default_device("cpu")

# Configurer les logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Charger les données depuis MongoDB
def load_data():
    from data_handling.data_loader import load_data_from_mongodb
    data = list(load_data_from_mongodb())
    return pd.DataFrame(data)

# Prétraiter les données
def preprocess_data(df):
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    # Vérifier les colonnes nécessaires
    required_columns = ["cleaned_text", "sentiment"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Colonnes manquantes : {', '.join(missing_columns)}")

    # Encodage des labels
    label_encoder = LabelEncoder()
    df["labels"] = label_encoder.fit_transform(df["sentiment"])
    return df, label_encoder

logger.info("Chargement des données depuis MongoDB...")
df = load_data()
df, label_encoder = preprocess_data(df)

# Créer un Dataset Hugging Face
dataset = Dataset.from_pandas(df)

# Charger le tokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# Tokeniser les données
def tokenize(batch):
    tokens = tokenizer(batch["cleaned_text"], truncation=True, padding=True)
    tokens["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)  # Labels en int64
    return tokens

logger.info("Tokenisation des données...")
dataset = dataset.map(tokenize, batched=True)

# Diviser en train et validation
logger.info("Division des données en ensembles d'entraînement et de validation...")
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
val_dataset = train_test_split["test"]

# Charger le modèle
num_labels = len(label_encoder.classes_)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)

# Configurer l'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
)

# Configurer Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Entraîner
logger.info("Début de l'entraînement du modèle...")
trainer.train()
logger.info("Entraînement terminé avec succès.")

# Évaluation
logger.info("Évaluation du modèle sur l'ensemble de validation...")
eval_results = trainer.evaluate()
logger.info(f"Résultats d'évaluation : {eval_results}")
