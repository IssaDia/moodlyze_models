import os
from pathlib import Path
import sys
import pandas as pd
import torch
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
import numpy as np
import logging
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': (predictions == labels).mean(),
    }

def train_bert_classifier(
    data,
    text_column='cleaned_text',
    label_column='sentiment',
    model_name="bert-base-uncased",
    batch_size=16,
    num_epochs=3,
    learning_rate=2e-5,
    max_length=128,
):
    try:
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Split data
        X = data[text_column].tolist()
        y = data[label_column].tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Initialize tokenizer and model
        logger.info("Initializing BERT model and tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(set(y)),
        ).to(device)

        # Encode data
        logger.info("Encoding training data...")
        train_encodings = tokenizer(
            X_train,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        logger.info("Encoding test data...")
        test_encodings = tokenizer(
            X_test,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # Encode labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)

        # Create datasets
        train_dataset = SentimentDataset(train_encodings, y_train_encoded)
        test_dataset = SentimentDataset(test_encodings, y_test_encoded)

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        # Train model
        logger.info("Starting training...")
        trainer.train()

        # Evaluate model
        logger.info("Evaluating model...")
        eval_result = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_result}")

        # Generate predictions and classification report
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test_encoded, y_pred))

        return model, tokenizer, le

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    from data_handling.data_loader import load_data_from_mongodb
    
    # Load your data
    raw_data = pd.DataFrame(list(load_data_from_mongodb()))
    
    # Train the model
    model, tokenizer, label_encoder = train_bert_classifier(raw_data)