"""Training and Evaluation Script for Conventional Baseline Classifiers.
Trains two separate baselines for rigorous empirical comparison:
1. Article-level Fake News Baseline: TF-IDF + Logistic Regression (REAL vs FAKE)
2. Claim-level FEVER Baseline: TF-IDF + Logistic Regression on claim text (SUPPORTED vs REFUTED vs UNCERTAIN)
"""

import json
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BASELINE_DATASET_PATH,
    BASELINE_MODEL_PATH,
    FEVER_BASELINE_MODEL_PATH,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    VALID_LABELS
)
from src.evaluation.baseline_classifier import BaselineClassifier
from src.utils.logging_utils import get_logger

logger = get_logger("train_baseline")

def train_and_evaluate_article_baseline():
    """Trains baseline on binary Real vs Fake articles dataset."""
    if not BASELINE_DATASET_PATH.exists():
        logger.error(f"Baseline dataset not found: {BASELINE_DATASET_PATH}")
        return None

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

    logger.info(f"Article Baseline Split sizes: Train={len(train_texts)}, Test={len(test_texts)}")

    classifier = BaselineClassifier()
    classifier.train(train_texts, train_labels)

    # Save trained checkpoint
    classifier.save(BASELINE_MODEL_PATH)

    # Evaluate on held-out test set
    metrics = classifier.evaluate(test_texts, test_labels)
    logger.info(f"Article Baseline Results on Test Split: Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}")

    # Persist report
    out_path = REPORTS_DIR / "baseline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Article baseline metrics report written to: {out_path}")
    return metrics

def train_and_evaluate_fever_baseline():
    """Trains surface-level TF-IDF baseline directly on FEVER claim verification dataset."""
    fever_path = PROCESSED_DATA_DIR / "fever_processed.jsonl"
    if not fever_path.exists():
        logger.error(f"FEVER processed dataset not found: {fever_path}")
        return None

    claims = []
    labels = []
    with open(fever_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                claims.append(item["claim"])
                labels.append(item["label"])

    logger.info(f"Loaded {len(claims)} FEVER claims from {fever_path}")

    # Stratified train/test split (70% train, 30% test)
    train_claims, test_claims, train_labels, test_labels = train_test_split(
        claims, labels, test_size=0.30, random_state=42, stratify=labels
    )

    logger.info(f"FEVER Baseline Split sizes: Train={len(train_claims)}, Test={len(test_claims)}")

    classifier = BaselineClassifier()
    classifier.train(train_claims, train_labels)

    # Save trained checkpoint
    classifier.save(FEVER_BASELINE_MODEL_PATH)

    # Evaluate on held-out test split
    metrics = classifier.evaluate(test_claims, test_labels)
    metrics["model_type"] = "TF-IDF + Logistic Regression (Surface-Level FEVER Baseline)"
    metrics["eval_dataset"] = "fever_processed.jsonl (Held-out Test Split)"
    logger.info(f"FEVER Baseline Results on Test Split: Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}")

    # Persist report
    out_path = REPORTS_DIR / "fever_baseline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"FEVER baseline metrics report written to: {out_path}")
    return metrics

def main():
    print("=" * 60)
    print("TRAINING CONVENTIONAL BASELINES")
    print("=" * 60)
    print("\n1. Training Article-level Binary Baseline (REAL vs FAKE)...")
    art_m = train_and_evaluate_article_baseline()
    print("\n2. Training Claim-level FEVER Baseline (SUPPORTED vs REFUTED vs UNCERTAIN)...")
    fev_m = train_and_evaluate_fever_baseline()
    print("=" * 60)
    print("BASELINE TRAINING COMPLETE")
    if art_m:
        print(f"Article Baseline Accuracy: {art_m['accuracy']:.4f} | Macro-F1: {art_m['macro_f1']:.4f}")
    if fev_m:
        print(f"FEVER Baseline Accuracy:   {fev_m['accuracy']:.4f} | Macro-F1: {fev_m['macro_f1']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
