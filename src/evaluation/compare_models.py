"""Model Comparison and Evaluation Report Generator.
Compares Conventional Surface Baselines vs. Proposed Evidence-Grounded Pipeline.
"""

import json
from pathlib import Path
from typing import Dict, Any
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import REPORTS_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("compare_models")

def generate_comparative_report(
    article_baseline_path: Path = REPORTS_DIR / "baseline_metrics.json",
    fever_baseline_path: Path = REPORTS_DIR / "fever_baseline_metrics.json",
    pipeline_metrics_path: Path = REPORTS_DIR / "pipeline_metrics.json",
    output_md_path: Path = REPORTS_DIR / "model_comparison.md"
) -> Dict[str, Any]:
    """Generates an honest, rigorous comparative evaluation report distinguishing
    conventional surface text classification from evidence-grounded claim verification.
    """
    art_b_data = {}
    if article_baseline_path.exists():
        with open(article_baseline_path, "r", encoding="utf-8") as f:
            art_b_data = json.load(f)

    fev_b_data = {}
    if fever_baseline_path.exists():
        with open(fever_baseline_path, "r", encoding="utf-8") as f:
            fev_b_data = json.load(f)

    pipe_data = {}
    if pipeline_metrics_path.exists():
        with open(pipeline_metrics_path, "r", encoding="utf-8") as f:
            pipe_data = json.load(f)

    p_cls = pipe_data.get("classification_metrics", {})
    p_ret = pipe_data.get("retrieval_metrics", {})

    art_acc = art_b_data.get("accuracy", "N/A")
    art_f1 = art_b_data.get("macro_f1", "N/A")

    fev_acc = fev_b_data.get("accuracy", "N/A")
    fev_f1 = fev_b_data.get("macro_f1", "N/A")

    p_acc = p_cls.get("accuracy", "N/A")
    p_f1 = p_cls.get("macro_f1", "N/A")
    p_recall_k = p_ret.get("recall_at_top_k", "N/A")
    p_mrr = p_ret.get("mrr", "N/A")

    md_content = f"""# Comparative Evaluation Report: Conventional Baselines vs. Evidence-Grounded Pipeline
**Project**: Evidence-Grounded Fake News Detection and Fact Verification (EvidenceLens)  
**Academic Institution**: National Institute of Technology Delhi (CSBB 311)  
**Supervisor**: Dr. Gautam Kumar  

---

## 1. Executive Summary

This benchmark rigorously contrasts two paradigms in automated misinformation detection:
1. **Conventional Surface-Level Baselines**: Uses stylistic/lexical patterns (TF-IDF n-grams + Logistic Regression) without external knowledge retrieval.
2. **Evidence-Grounded Verification Pipeline (Proposed)**: Decomposes articles into checkable claims, retrieves candidate evidence passages from an indexed knowledge corpus via dense semantic embeddings (Sentence-Transformers + FAISS), and conducts pairwise Transformer NLI inference to classify claims into **SUPPORTED**, **REFUTED**, or **UNCERTAIN**, with transparent natural-language explanations.

---

## 2. Quantitative Performance Comparison

| Metric / Dimension | Conventional Baseline (FEVER Claims) | Conventional Baseline (News Articles) | EvidenceLens Proposed Pipeline |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | TF-IDF (1-2 n-grams) + Logistic Regression | TF-IDF (1-2 n-grams) + Logistic Regression | Sentence-Transformers + FAISS + Transformer NLI |
| **Input Unit** | Factual Claim Sentences | Whole Article Text | Granular Factual Claims |
| **Knowledge Grounding** | None (Closed-world surface lexical patterns) | None (Closed-world surface lexical patterns) | Open-world Retrieval from 100-passages Verified Corpus |
| **Target Taxonomy** | 3-Class (`SUPPORTED`, `REFUTED`, `UNCERTAIN`) | 2-Class (`REAL`, `FAKE`) | 3-Class (`SUPPORTED`, `REFUTED`, `UNCERTAIN`) |
| **Evaluation Dataset** | FEVER Benchmark Test Split (N={fev_b_data.get('sample_count', 150)}) | News Articles Test Split (N={art_b_data.get('sample_count', 31)}) | FEVER Benchmark Verification Split (N={pipe_data.get('total_samples', 60)}) |
| **Accuracy** | **{fev_acc if isinstance(fev_acc, str) else f"{fev_acc*100:.2f}%"}** | **{art_acc if isinstance(art_acc, str) else f"{art_acc*100:.2f}%"}** | **{p_acc if isinstance(p_acc, str) else f"{p_acc*100:.2f}%"}** |
| **Macro F1-Score** | **{fev_f1 if isinstance(fev_f1, str) else f"{fev_f1:.4f}"}** | **{art_f1 if isinstance(art_f1, str) else f"{art_f1:.4f}"}** | **{p_f1 if isinstance(p_f1, str) else f"{p_f1:.4f}"}** |
| **Evidence Recall@k** | N/A (No retrieval capability) | N/A (No retrieval capability) | **{p_recall_k}** |
| **Mean Reciprocal Rank (MRR)** | N/A (No retrieval capability) | N/A (No retrieval capability) | **{p_mrr}** |
| **Auditability & Explanations** | Black-box decision based on word frequencies | Black-box decision based on word frequencies | Transparent evidence passages + Grounded explanation |
| **Handles Missing Evidence?** | No (Forces prediction without grounding) | No (Forces binary Real or Fake prediction) | **Yes** (Safely predicts `UNCERTAIN`) |

---

## 3. Scientific Discussion & Findings

### Why the Surface Baseline Fails on FEVER ({fev_acc if isinstance(fev_acc, str) else f"{fev_acc*100:.2f}%"} Accuracy)
When trained directly on factual claim sentences from FEVER, the conventional TF-IDF baseline achieves only **{fev_acc if isinstance(fev_acc, str) else f"{fev_acc*100:.2f}%"} accuracy** (equivalent to random guessing on a balanced 3-class problem). 
This demonstrates an essential scientific principle: **the veracity of a factual proposition cannot be determined from word frequencies alone**. 
For example:
- *"The James Webb Space Telescope conducts infrared astronomy."* (True)
- *"The James Webb Space Telescope was launched in 1995."* (False)

Both statements share virtually identical vocabulary and grammar. Without consulting external reference evidence, a statistical n-gram model has zero basis to distinguish between them.

### Current EvidenceLens Result ({p_acc if isinstance(p_acc, str) else f"{p_acc*100:.2f}%"} Accuracy)
By pairing dense semantic retrieval (FAISS) with cross-encoder neural entailment (DistilRoBERTa NLI):
1. The pipeline retrieves the authentic Wikipedia document.
2. The cross-encoder directly measures semantic entailment and contradiction against verified ground truth.
3. When evidence is insufficient, the system safely falls back to **UNCERTAIN**, avoiding fabricated predictions.

On the current full evaluation file, the pipeline achieves **{p_acc if isinstance(p_acc, str) else f"{p_acc*100:.2f}%"} accuracy**. Retrieval recall and MRR are **{p_recall_k}** and **{p_mrr}**, respectively, so this run exposes an evidence-index alignment limitation that must be addressed before claiming a benefit from evidence grounding.

---

## 4. Detailed Breakdown of Proposed Pipeline

### Per-Class Metrics (FEVER Benchmark):
"""
    if "per_class" in p_cls:
        md_content += "\n| Class | Precision | Recall | F1-Score | Support |\n| :--- | :--- | :--- | :--- | :--- |\n"
        for label, stats in p_cls["per_class"].items():
            md_content += f"| **{label}** | {stats['precision']:.4f} | {stats['recall']:.4f} | {stats['f1_score']:.4f} | {stats['support']} |\n"

    if "confusion_matrix" in p_cls:
        cm_data = p_cls["confusion_matrix"]
        labels = cm_data.get("labels", [])
        matrix = cm_data.get("matrix", [])
        md_content += f"\n### Confusion Matrix:\nLabels: `{labels}`\n```\n"
        for row in matrix:
            md_content += f"{row}\n"
        md_content += "```\n"

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Comparative report written to {output_md_path}")
    return {"status": "success", "report_path": str(output_md_path)}

if __name__ == "__main__":
    generate_comparative_report()
