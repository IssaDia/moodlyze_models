from transformers import BertTokenizer, BertForSequenceClassification
import torch
from pathlib import Path
from enum import Enum
import torch.nn.functional as F
from typing import Dict, Any
from .config import ModelType, get_model_config
from utils.exceptions import ModelLoadError
import joblib

class SentimentAnalyzer:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        """Load the appropriate model based on model_type"""
        config = get_model_config(self.model_type)
        
        if self.model_type == ModelType.BERT:
            try:
                model_path = Path(config.model_path)
                
                # Load model
                checkpoint = torch.load(
                    str(model_path), 
                    map_location=self.device
                )
                
                self.model = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased",
                    num_labels=3,
                    output_attentions=False,
                    output_hidden_states=False
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                # Load tokenizer
                self.tokenizer = BertTokenizer.from_pretrained(
                    str(model_path / "tokenizer")
                )
                
            except Exception as e:
                raise ModelLoadError(f"Error loading BERT model: {str(e)}")
        else:
            # Handle other model types (Naive Bayes, etc.)
            try:
                self.model = joblib.load(config.model_path)
                self.vectorizer = joblib.load(config.vectorizer_path)
            except FileNotFoundError:
                raise ModelLoadError(f"Model or vectorizer file not found for {config.name}")
            except Exception as e:
                raise ModelLoadError(f"Error loading {config.name}: {str(e)}")

    def predict(self, text: str) -> str:
        """Predict sentiment for given text"""
        if self.model_type == ModelType.BERT:
            if self.model is None or self.tokenizer is None:
                raise ModelLoadError("BERT model or tokenizer not loaded")
            
            try:
                # Tokenize input
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors="pt"
                )
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get prediction
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = F.softmax(outputs.logits, dim=1)
                    prediction = torch.argmax(probabilities, dim=1).item()
                
                # Map prediction to sentiment
                sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
                return sentiment_map[prediction]
                
            except Exception as e:
                raise ValueError(f"Error predicting with BERT: {str(e)}")
        else:
            # Original logic for other models
            if self.vectorizer is None:
                raise ModelLoadError("Vectorizer not loaded")
            
            vectorized_text = self.vectorizer.transform([text])
            try:
                prediction = self.model.predict(vectorized_text)[0]
                
                # Map prediction to sentiment
                if prediction == "negative":
                    return 'Negative'
                elif prediction == "neutral":
                    return 'Neutral'
                elif prediction == "positive":
                    return 'Positive'
                else:
                    raise ValueError("Unexpected prediction value")
            except Exception as e:
                raise ValueError(f"Error predicting: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            "model_type": self.model_type.value,
            "model_class": type(self.model).__name__,
        }
        
        if self.model_type == ModelType.BERT:
            info.update({
                "model_architecture": "BERT",
                "max_sequence_length": 128,
                "num_labels": 3,
                "device": str(self.device)
            })
        else:
            info.update({
                "vectorizer_vocab_size": len(self.vectorizer.vocabulary_),
                "sample_vocab": list(self.vectorizer.vocabulary_.items())[:5]
            })
            
        return info