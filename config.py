from enum import Enum
from dataclasses import dataclass
import os

class ModelType(Enum):
    NAIVE_BAYES = "naive_bayes"
    LOGISTIC_REGRESSION = "logistic_regression"
    BERT = "bert"

@dataclass
class ModelConfig:
    name: str
    model_path: str
    tokenizer_path: str

def get_model_config(model_type: ModelType) -> ModelConfig:
    base_path = os.path.join(os.path.dirname(__file__), "saved_models")
    print(f"Base path: {base_path}") 
    configs = {
        ModelType.NAIVE_BAYES: ModelConfig(
            name="Naive Bayes",
            model_path=os.path.join(base_path, "naive_bayes", "naive_bayes.pkl"),
            tokenizer_path=None
        ),
        ModelType.LOGISTIC_REGRESSION: ModelConfig(
            name="Logistic Regression",
            model_path=os.path.join(base_path, "logistic_regression", "logistic_regression.pkl"),
            tokenizer_path=os.path.join(base_path, "logistic_regression", "vectorizer_lr.pkl")
        ),
         ModelType.BERT: ModelConfig(
            name="bert",
            model_path=os.path.join(base_path, "bert", "bert.pkl"),
            tokenizer_path=os.path.join(base_path, "bert", "vectorizer_bert.pkl")
        )
    }
    return configs[model_type]