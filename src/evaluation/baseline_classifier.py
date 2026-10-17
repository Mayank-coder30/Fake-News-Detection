import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from config import BASELINE_MODEL_PATH
from src.utils.logging_utils import get_logger

logger = get_logger("BaselineClassifier")

class BaselineClassifier:
    """Conventional text-classification baseline using TF-IDF feature extraction and Logistic Regression."""

    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words="english",
                sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=500,
                random_state=42
            ))
        ])
        self.classes_ = None

    def train(self, texts: List[str], labels: List[str]):
        """Trains the TF-IDF + Logistic Regression pipeline."""
        logger.info(f"Training baseline classifier on {len(texts)} samples...")
        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.classes_)
        logger.info(f"Model trained successfully. Classes: {self.classes_}")

    def predict(self, texts: List[str]) -> List[str]:
        """Predicts class labels for text inputs."""
        return list(self.pipeline.predict(texts))

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns predicted class probabilities."""
        return self.pipeline.predict_proba(texts)

    def evaluate(self, texts: List[str], gold_labels: List[str]) -> Dict[str, Any]:
        """Evaluates model performance against gold standard labels on held-out split."""
        preds = self.predict(texts)
        acc = float(accuracy_score(gold_labels, preds))
        prec, rec, f1, _ = precision_recall_fscore_support(gold_labels, preds, average="macro", zero_division=0)
        
        # Per-class metrics
        per_class_p, per_class_r, per_class_f, support = precision_recall_fscore_support(
            gold_labels, preds, labels=self.classes_, zero_division=0
        )
        cm = confusion_matrix(gold_labels, preds, labels=self.classes_).tolist()

        metrics = {
            "model_type": "TF-IDF + Logistic Regression (Conventional Text Classifier)",
            "sample_count": len(texts),
            "classes": self.classes_,
            "accuracy": round(acc, 4),
            "macro_precision": round(float(prec), 4),
            "macro_recall": round(float(rec), 4),
            "macro_f1": round(float(f1), 4),
            "per_class": {
                cls_name: {
                    "precision": round(float(per_class_p[i]), 4),
                    "recall": round(float(per_class_r[i]), 4),
                    "f1_score": round(float(per_class_f[i]), 4),
                    "support": int(support[i])
                }
                for i, cls_name in enumerate(self.classes_)
            },
            "confusion_matrix": cm
        }
        return metrics

    def save(self, path: Path = BASELINE_MODEL_PATH):
        """Saves trained pipeline to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, str(path))
        logger.info(f"Saved baseline classifier to: {path}")

    @classmethod
    def load(cls, path: Path = BASELINE_MODEL_PATH) -> "BaselineClassifier":
        """Loads serialized pipeline from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Baseline model checkpoint not found: {path}")
        inst = cls()
        inst.pipeline = joblib.load(str(path))
        inst.classes_ = list(inst.pipeline.classes_)
        logger.info(f"Loaded baseline classifier from: {path}")
        return inst
