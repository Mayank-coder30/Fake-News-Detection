import json
from pathlib import Path
from typing import Dict, Any

from config import REPORTS_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("compare_models")

def generate_comparative_report(
    baseline_metrics_path: Path = REPORTS_DIR / "baseline_metrics.json",
    pipeline_metrics_path: Path = REPORTS_DIR / "pipeline_metrics.json",
    output_md_path: Path = REPORTS_DIR / "model_comparison.md"
) -> Dict[str, Any]:
    """Generates an honest, rigorous comparative evaluation report distinguishing
    conventional text classification from evidence-grounded claim verification.
    """
    baseline_data = {}
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)

    pipeline_data = {}
    if pipeline_metrics_path.exists():
        with open(pipeline_metrics_path, "r", encoding="utf-8") as f:
            pipeline_data = json.load(f)

    p_cls = pipeline_data.get("classification_metrics", {})
    p_ret = pipeline_data.get("retrieval_metrics", {})

    b_acc = baseline_data.get("accuracy", "NOT RUN")
    b_f1 = baseline_data.get("macro_f1", "NOT RUN")
    p_acc = p_cls.get("accuracy", "NOT RUN")
    p_f1 = p_cls.get("macro_f1", "NOT RUN")
    p_recall_k = p_ret.get("recall_at_top_k", "NOT RUN")
    p_mrr = p_ret.get("mrr", "NOT RUN")

    md_content = f"""# Comparative Evaluation Report: Conventional Baseline vs. Evidence-Grounded Pipeline
**Project**: Evidence-Grounded Fake News Detection and Fact Verification (EvidenceLens)  
**Academic Institution**: National Institute of Technology Delhi (CSBB 311)

---

## 1. Executive Summary

This evaluation contrasts two fundamentally distinct paradigms in automated misinformation detection:
1. **Conventional Text Classification Baseline**: Uses stylistic/lexical patterns (TF-IDF n-grams + Logistic Regression) to assign coarse binary Real/Fake labels without external verification.
2. **Evidence-Grounded Verification Pipeline (Proposed)**: Decomposes articles into checkable claims, retrieves candidate evidence passages from a indexed knowledge corpus via dense semantic embeddings (Sentence-Transformers + FAISS), and conducts pairwise Transformer NLI inference to classify claims into **SUPPORTED**, **REFUTED**, or **UNCERTAIN**, with transparent natural-language explanations.

---

## 2. Quantitative Performance Comparison

| Metric / Attribute | Conventional Text Classifier (Baseline) | Evidence-Grounded Pipeline (Proposed) |
| :--- | :--- | :--- |
| **Model Architecture** | TF-IDF (1-2 ngrams) + Logistic Regression | Sentence-Transformers + FAISS + Transformer NLI |
| **Input Unit** | Entire news article / headline | Granular, individual factual claims |
| **Knowledge Grounding** | None (Closed-world surface lexical patterns) | Open-world Retrieval from Verified Knowledge Corpus |
| **Target Taxonomy** | 2-Class (`REAL`, `FAKE`) | 3-Class (`SUPPORTED`, `REFUTED`, `UNCERTAIN`) |
| **Evaluation Dataset** | News Articles Held-out Split (N={baseline_data.get('sample_count', 'N/A')}) | FEVER Benchmark Verification Split (N={pipeline_data.get('total_samples', 'N/A')}) |
| **Accuracy** | **{b_acc}** | **{p_acc}** |
| **Macro F1-Score** | **{b_f1}** | **{p_f1}** |
| **Evidence Recall@k** | N/A (No retrieval capability) | **{p_recall_k}** |
| **Mean Reciprocal Rank (MRR)** | N/A (No retrieval capability) | **{p_mrr}** |
| **Auditability / Interpretability** | Black-box decision based on word frequencies | Transparent evidence passages + Grounded explanation |
| **Handles Missing Evidence?** | No (Forces binary Real or Fake prediction) | **Yes** (Safely predicts `UNCERTAIN`) |

---

## 3. Methodological Distinctions & Discussion

### A. Why Conventional Classification Fails in Practice
Conventional classifiers detect superficial stylistic proxies (sensationalist words, punctuation, emotional rhetoric). When applied to plausible, professionally written misinformation or novel breaking news, surface text classifiers suffer severe distribution shift and fail to determine whether the factual claims are true.

### B. The Advantage of the Evidence-Grounded Approach
By grounding each claim in retrieved evidence passages:
1. **Verifiable Audit Trail**: Journalists, editors, and students can review the exact source passage and relevance score.
2. **Epistemic Modality (UNCERTAIN)**: Distinguishes between claims that are actively refuted by contradictory evidence versus claims where evidence is currently unavailable or inconclusive.
3. **No Fabricated Labels**: A low retrieval score triggers `UNCERTAIN` rather than false certainty.

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
