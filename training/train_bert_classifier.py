from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import sys
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import shutil

# Add parent directory to system path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_handling.data_loader import load_data_from_mongodb
from nlp.sentiment_analysis import analyze_sentiment

class SentimentDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]).clone().detach() for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])

def setup_model_paths():
    """Configure and verify model save paths"""
    # Get absolute path to project root
    current_dir = Path(os.getcwd())
    
    # Setup paths
    model_dir = current_dir /  "models" / "saved_models" / "bert"
    model_path = model_dir / "model"
    
    # Create directories with explicit permissions
    os.makedirs(model_dir, mode=0o755, exist_ok=True)
    os.makedirs(model_path, mode=0o755, exist_ok=True)
    
    return model_dir, model_path

def verify_save_directory(directory):
    """Verify write permissions in save directory with detailed checks"""
    directory = Path(directory)
    
    # Check if directory exists and is writable
    if not directory.exists():
        print(f"Directory does not exist: {directory}")
        return False
        
    if not os.access(directory, os.W_OK):
        print(f"No write permission for directory: {directory}")
        return False
    
    # Test file creation and deletion
    test_file = directory / "write_test.txt"
    try:
        test_file.write_text("test")
        test_file.unlink()
        return True
    except Exception as e:
        print(f"Directory verification failed: {str(e)}")
        print(f"Current permissions: {oct(os.stat(directory).st_mode)[-3:]}")
        print(f"Current owner: {os.stat(directory).st_uid}")
        return False

def save_model_with_torch(model, tokenizer, save_path):
    """Save model and tokenizer using torch.save"""
    save_path = Path(save_path)
    print(f"\nAttempting to save model to: {save_path}")

    try:
        # Create directory if needed
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        print("Saving model...")
        model_save_path = save_path / "model_state.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': model.config,
        }, str(model_save_path))
        
        # Verify model save
        if not model_save_path.exists():
            raise FileNotFoundError(f"Model file not saved correctly at {model_save_path}")
        
        # Save tokenizer
        print("Saving tokenizer...")
        tokenizer_save_path = save_path / "tokenizer"
        tokenizer.save_pretrained(str(tokenizer_save_path))
        
        print("\nSaved files:")
        for file in save_path.rglob("*"):
            if file.is_file():
                size = file.stat().st_size
                print(f"- {file.relative_to(save_path)} ({size/1024:.2f} KB)")
        
        return True
        
    except Exception as e:
        print(f"\nError during save: {str(e)}")
        print(f"Attempted save path: {save_path}")
        print(f"Current working directory: {os.getcwd()}")
        return False

def load_model_with_torch(model_path, num_labels=3):
    """Load saved model and tokenizer"""
    model_path = Path(model_path)
    
    try:
        # Load model
        print("Loading model...")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        
        checkpoint = torch.load(str(model_path / "model_state.pt"))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(str(model_path / "tokenizer"))
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error during loading: {str(e)}")
        raise

def balance_classes(data, target_column):
    """Balance class distribution in the dataset"""
    classes = data[target_column].unique()
    class_counts = data[target_column].value_counts()
    min_count = class_counts.min()

    balanced_data = pd.DataFrame()
    for label in classes:
        class_subset = data[data[target_column] == label]
        balanced_subset = resample(class_subset, replace=True, n_samples=min_count, random_state=42)
        balanced_data = pd.concat([balanced_data, balanced_subset])

    return balanced_data.sample(frac=1, random_state=42)

def analyze_class_distribution(y, title="Class Distribution"):
    """Analyze and print class distribution"""
    distribution = pd.Series(y).value_counts()
    total = len(y)
    print(f"\n{title}:")
    print("=" * 50)
    for label, count in distribution.items():
        percentage = (count / total) * 100
        print(f"{label}: {count} ({percentage:.2f}%)")

def train_bert_classifier():
    try:
        # Setup paths and verify permissions
        model_dir, model_path = setup_model_paths()
        model_dir = Path(model_dir)
        model_path = Path(model_path)
        
        print(f"\nModel will be saved to: {model_path}")
        
        if not verify_save_directory(model_dir):
            raise PermissionError(f"No write permissions in {model_dir}")

        # Load and prepare data
        raw_data = pd.DataFrame(list(load_data_from_mongodb()))
        print(f"Number of elements in raw_data: {len(raw_data)}")

        # Verify required columns
        required_columns = ['cleaned_text', 'sentiment']
        missing_columns = [col for col in required_columns if col not in raw_data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        # Analyze class distribution
        print("\nInitial sentiment distribution:")
        analyze_class_distribution(raw_data['sentiment'])

        # Balance classes
        balanced_data = balance_classes(raw_data, 'sentiment')
        print("\nDistribution after class balancing:")
        analyze_class_distribution(balanced_data['sentiment'])

        # Prepare features and labels
        X = balanced_data["cleaned_text"].tolist()
        y = balanced_data["sentiment"].tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Initialize tokenizer and model
        print("\nInitializing BERT model and tokenizer...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=3,
            output_attentions=False,
            output_hidden_states=False
        )

        # Test save functionality
        print("\nTesting save functionality...")
        test_save = save_model_with_torch(model, tokenizer, model_path)
        if test_save:
            print("Save test successful")
            # Test loading
            loaded_model, loaded_tokenizer = load_model_with_torch(model_path)
            print("Load test successful")
        else:
            raise RuntimeError("Initial save test failed")

        # Prepare datasets
        train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
        test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

        # Encode labels
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        train_encodings["labels"] = torch.tensor(y_train)
        test_encodings["labels"] = torch.tensor(y_test)

        train_dataset = SentimentDataset(train_encodings)
        test_dataset = SentimentDataset(test_encodings)

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(model_path),
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_dir=str(model_path / "logs"),
            evaluation_strategy="epoch",
            save_strategy="no",  # Disable intermediate saving
            save_total_limit=None,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        # Train model
        print("\nStarting training...")
        trainer.train()

        # Evaluate model
        print("\nEvaluating model...")
        eval_result = trainer.evaluate()
        print(f"\nEvaluation results: {eval_result}")

        predictions = trainer.predict(test_dataset)
        y_pred = predictions.predictions.argmax(-1)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Save final model
        print("\nSaving final model...")
        if not save_model_with_torch(model, tokenizer, model_path):
            raise RuntimeError("Failed to save model and tokenizer")

        print(f"\nTraining completed and model saved successfully at: {model_path}")

    except Exception as e:
        print(f"\nError in training process: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        train_bert_classifier()
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)