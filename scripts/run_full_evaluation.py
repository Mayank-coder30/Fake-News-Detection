"""Master Independent Full-Dataset Benchmark Runner.
Runs both the baseline classifier and the EvidenceLens pipeline on the full FEVER dataset (19,998 records).
Saves intermediate checkpoints, incremental predictions, and comprehensive benchmark metrics.
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path
from collections import Counter

# Set root directory
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Ensure OpenMP stability on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import (
    RAW_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    VALID_LABELS,
    LABEL_SUPPORTED,
    LABEL_REFUTED,
    LABEL_UNCERTAIN
)
from src.evaluation.baseline_classifier import BaselineClassifier
from src.pipeline.fact_check_pipeline import FactCheckPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

LOG_FILE = REPORTS_DIR / "full_run.log"
CHECKPOINT_FILE = REPORTS_DIR / "full_pipeline_checkpoint.json"
PREDICTIONS_FILE = REPORTS_DIR / "full_pipeline_predictions.jsonl"
FINAL_PIPELINE_FILE = REPORTS_DIR / "full_pipeline_metrics.json"
FINAL_BASELINE_FILE = REPORTS_DIR / "full_baseline_metrics.json"

LABEL_MAPPING = {
    "SUPPORTS": LABEL_SUPPORTED,
    "REFUTES": LABEL_REFUTED,
    "NOT ENOUGH INFO": LABEL_UNCERTAIN
}

def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("STARTING FULL DATASET BENCHMARK EXECUTION")
    log("=" * 70)

    raw_file = RAW_DATA_DIR / "dev.jsonl"
    if not raw_file.exists():
        log(f"ERROR: Raw dataset not found at {raw_file}")
        return

    # 1. Load full dataset
    log(f"Loading full dataset from {raw_file}...")
    claims = []
    labels = []
    ids = []
    
    with open(raw_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                raw_lbl = rec.get("label", "").strip()
                claim_text = rec.get("claim", "").strip()
                if raw_lbl in LABEL_MAPPING and claim_text:
                    ids.append(rec.get("id"))
                    claims.append(claim_text)
                    labels.append(LABEL_MAPPING[raw_lbl])
            except Exception:
                continue

    total_records = len(claims)
    log(f"Total valid labeled records loaded: {total_records}")
    log(f"Label distribution: {dict(Counter(labels))}")

    # =========================================================================
    # STAGE 1: TRAIN & EVALUATE BASELINE CLASSIFIER ON FULL DATASET
    # =========================================================================
    log("\n" + "=" * 70)
    log("STAGE 1: TRAINING & EVALUATING BASELINE MODEL (TF-IDF + LOGISTIC REGRESSION)")
    log("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        claims, labels, test_size=0.20, random_state=42, stratify=labels
    )
    log(f"Baseline split: {len(X_train)} training claims, {len(X_test)} test claims (80/20 split)")

    t0_base = time.time()
    baseline = BaselineClassifier()
    baseline.train(X_train, y_train)
    base_train_time = round(time.time() - t0_base, 2)
    log(f"Baseline model trained in {base_train_time} seconds.")

    # Save trained baseline model
    base_model_path = MODELS_DIR / "full_baseline_classifier.joblib"
    baseline.save(base_model_path)
    log(f"Baseline model saved to: {base_model_path}")

    # Evaluate on held-out test split
    base_preds = baseline.predict(X_test)
    base_acc = float(accuracy_score(y_test, base_preds))
    base_macro_p, base_macro_r, base_macro_f1, _ = precision_recall_fscore_support(
        y_test, base_preds, labels=VALID_LABELS, average="macro", zero_division=0
    )
    base_p, base_r, base_f1, base_sup = precision_recall_fscore_support(
        y_test, base_preds, labels=VALID_LABELS, zero_division=0
    )
    base_cm = confusion_matrix(y_test, base_preds, labels=VALID_LABELS).tolist()

    base_metrics = {
        "model_type": "TF-IDF + Logistic Regression",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(base_acc, 4),
        "macro_precision": round(float(base_macro_p), 4),
        "macro_recall": round(float(base_macro_r), 4),
        "macro_f1": round(float(base_macro_f1), 4),
        "per_class": {
            lbl: {
                "precision": round(float(base_p[i]), 4),
                "recall": round(float(base_r[i]), 4),
                "f1_score": round(float(base_f1[i]), 4),
                "support": int(base_sup[i])
            }
            for i, lbl in enumerate(VALID_LABELS)
        },
        "confusion_matrix": base_cm
    }

    with open(FINAL_BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(base_metrics, f, indent=2)
    log(f"Baseline metrics written to: {FINAL_BASELINE_FILE}")
    log(f"Baseline Test Accuracy: {base_acc*100:.2f}% | Macro F1: {base_macro_f1:.4f}")

    # =========================================================================
    # STAGE 2: EVALUATE EVIDENCELENS PIPELINE ON FULL DATASET
    # =========================================================================
    log("\n" + "=" * 70)
    log("STAGE 2: EVALUATING EVIDENCELENS PIPELINE (FAISS + CROSS-ENCODER NLI)")
    log("=" * 70)

    log("Initializing FactCheckPipeline...")
    t0_pipe_init = time.time()
    pipeline = FactCheckPipeline()
    log(f"FactCheckPipeline initialized in {time.time() - t0_pipe_init:.2f}s.")

    # Check for existing checkpoint to resume gracefully if restarted
    start_idx = 0
    gold_history = []
    pred_history = []

    if CHECKPOINT_FILE.exists():
        try:
            ckpt = json.load(open(CHECKPOINT_FILE, "r", encoding="utf-8"))
            start_idx = ckpt.get("completed_count", 0)
            gold_history = ckpt.get("gold_history", [])
            pred_history = ckpt.get("pred_history", [])
            log(f"Resuming from checkpoint at record {start_idx} / {total_records}...")
        except Exception as e:
            log(f"Could not load checkpoint: {e}. Starting fresh.")
            start_idx = 0
            gold_history = []
            pred_history = []

    pipe_start_time = time.time()
    last_checkpoint_time = time.time()

    for idx in range(start_idx, total_records):
        cid = str(ids[idx])
        claim = claims[idx]
        gold = labels[idx]

        try:
            passages = pipeline.retriever.retrieve_for_claim(claim, top_k=3)
            res = pipeline.verifier.verify(cid, claim, passages)
            pred = res.label
        except Exception as e:
            pred = LABEL_UNCERTAIN
            log(f"Warning: error on claim {cid}: {e}")

        gold_history.append(gold)
        pred_history.append(pred)

        # Append prediction line
        with open(PREDICTIONS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": cid,
                "claim": claim,
                "gold": gold,
                "pred": pred,
                "correct": bool(gold == pred)
            }) + "\n")

        processed_so_far = len(gold_history)

        # Checkpoint and log progress every 250 claims
        if processed_so_far % 250 == 0 or processed_so_far == total_records:
            cur_acc = accuracy_score(gold_history, pred_history)
            elapsed = time.time() - pipe_start_time
            rate = (processed_so_far - start_idx) / max(elapsed, 0.001)
            remaining = total_records - processed_so_far
            eta_mins = (remaining / max(rate, 0.001)) / 60

            log(
                f"Progress: {processed_so_far}/{total_records} ({processed_so_far*100/total_records:.1f}%) | "
                f"Rate: {rate:.1f} claims/s | Running Acc: {cur_acc*100:.2f}% | ETA: {eta_mins:.1f}m"
            )

            # Save checkpoint
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "completed_count": processed_so_far,
                    "total_records": total_records,
                    "running_accuracy": round(float(cur_acc), 4),
                    "gold_history": gold_history,
                    "pred_history": pred_history,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f)

    # Final Pipeline Metrics Computation
    pipe_elapsed_total = round(time.time() - pipe_start_time, 2)
    final_acc = float(accuracy_score(gold_history, pred_history))
    pipe_macro_p, pipe_macro_r, pipe_macro_f1, _ = precision_recall_fscore_support(
        gold_history, pred_history, labels=VALID_LABELS, average="macro", zero_division=0
    )
    pipe_p, pipe_r, pipe_f1, pipe_sup = precision_recall_fscore_support(
        gold_history, pred_history, labels=VALID_LABELS, zero_division=0
    )
    pipe_cm = confusion_matrix(gold_history, pred_history, labels=VALID_LABELS).tolist()

    pipeline_metrics = {
        "pipeline_name": "EvidenceLens Fact Verification Pipeline",
        "eval_dataset": "dev.jsonl (Full 19,998 Records)",
        "total_samples": total_records,
        "elapsed_seconds": pipe_elapsed_total,
        "classification_metrics": {
            "accuracy": round(final_acc, 4),
            "macro_precision": round(float(pipe_macro_p), 4),
            "macro_recall": round(float(pipe_macro_r), 4),
            "macro_f1": round(float(pipe_macro_f1), 4),
            "per_class": {
                lbl: {
                    "precision": round(float(pipe_p[i]), 4),
                    "recall": round(float(pipe_r[i]), 4),
                    "f1_score": round(float(pipe_f1[i]), 4),
                    "support": int(pipe_sup[i])
                }
                for i, lbl in enumerate(VALID_LABELS)
            },
            "confusion_matrix": {
                "labels": VALID_LABELS,
                "matrix": pipe_cm
            }
        }
    }

    with open(FINAL_PIPELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(pipeline_metrics, f, indent=2)

    log("\n" + "=" * 70)
    log("FULL BENCHMARK COMPLETE!")
    log("=" * 70)
    log(f"Baseline Accuracy (4,000 Test Claims):       {base_acc*100:.2f}% (Macro-F1: {base_macro_f1:.4f})")
    log(f"EvidenceLens Accuracy (19,998 Claims):       {final_acc*100:.2f}% (Macro-F1: {pipe_macro_f1:.4f})")
    log(f"All artifacts written to: {REPORTS_DIR}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR in execution: {e}")
        log(traceback.format_exc())

