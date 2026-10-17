"""Training and Evaluation Script for Conventional Fake News Classifier Baseline.
"""

import json
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BASELINE_DATASET_PATH, BASELINE_MODEL_PATH, REPORTS_DIR
from src.evaluation.baseline_classifier import BaselineClassifier
from src.utils.logging_utils import get_logger

logger = get_logger("train_baseline")

def train_and_evaluate_baseline():
    if not BASELINE_DATASET_PATH.exists():
        logger.error(f"Baseline dataset not found: {BASELINE_DATASET_PATH}")
        return

    texts = []
    labels = []
    with open(BASELINE_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                texts.append(item["text"])
                labels.append(item["label"])

    logger.info(f"Loaded {len(texts)} samples from {BASELINE_DATASET_PATH}")

    # Stratified train/test split (70% train, 30% test)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.30, random_state=42, stratify=labels
    )

    logger.info(f"Split sizes: Train={len(train_texts)}, Test={len(test_texts)}")

    classifier = BaselineClassifier()
    classifier.train(train_texts, train_labels)

    # Save trained checkpoint
    classifier.save(BASELINE_MODEL_PATH)

    # Evaluate on held-out test set
    metrics = classifier.evaluate(test_texts, test_labels)
    logger.info(f"Evaluation Results on Test Split: Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}")

    # Persist report
    out_path = REPORTS_DIR / "baseline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Baseline metrics report written to: {out_path}")
    return metrics

if __name__ == "__main__":
    train_and_evaluate_baseline()
