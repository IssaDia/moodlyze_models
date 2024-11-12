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
    vectorizer_path: str

def get_model_config(model_type: ModelType) -> ModelConfig:
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_path = os.path.join(project_dir, "models", "saved_models")
    configs = {
        ModelType.NAIVE_BAYES: ModelConfig(
            name="Naive Bayes",
            model_path=os.path.join(base_path, "naive_bayes", "naive_bayes.pkl"),
            vectorizer_path=os.path.join(base_path, "naive_bayes", "vectorizer_nb.pkl")
        ),
        ModelType.LOGISTIC_REGRESSION: ModelConfig(
            name="Logistic Regression",
            model_path=os.path.join(base_path, "logistic_regression", "logistic_regression.pkl"),
            vectorizer_path=os.path.join(base_path, "logistic_regression", "vectorizer_lr.pkl")
        ),
         ModelType.BERT: ModelConfig(
            name="Bert",
            model_path=os.path.join(base_path, "bert", "model_state.pt"),
            vectorizer_path=os.path.join(base_path, "bert", "tokenizer")
        ),
    }
    return configs[model_type]