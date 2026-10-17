import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    LABEL_SUPPORTED,
    LABEL_REFUTED,
    LABEL_UNCERTAIN,
    VALID_LABELS
)
from src.pipeline.fact_check_pipeline import FactCheckPipeline
from src.utils.logging_utils import get_logger

logger = get_logger("evaluate_verifier")

class VerifierEvaluator:
    """Evaluates the end-to-end evidence retrieval and claim-verification pipeline."""

    def __init__(self, pipeline: FactCheckPipeline):
        self.pipeline = pipeline

    def evaluate_on_fever(self, fever_path: Path) -> Dict[str, Any]:
        """Runs the pipeline over processed FEVER records and computes standard evaluation metrics."""
        if not fever_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {fever_path}")

        records = []
        with open(fever_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        logger.info(f"Evaluating pipeline on {len(records)} FEVER test records...")

        gold_labels = []
        predicted_labels = []
        retrieval_hits = 0
        mrr_total = 0.0
        claims_with_evidence_count = 0

        detailed_results = []

        for rec in records:
            claim_id = str(rec["id"])
            claim_text = rec["claim"]
            gold = rec["label"]
            evidence_refs = [ref["doc_title"] for ref in rec.get("evidence_refs", [])]

            # Run retrieval & verification
            passages = self.pipeline.retriever.retrieve_for_claim(claim_text, top_k=self.pipeline.top_k)
            v_res = self.pipeline.verifier.verify(claim_id, claim_text, passages)

            pred = v_res.label
            gold_labels.append(gold)
            predicted_labels.append(pred)

            # Retrieval metric calculation
            if evidence_refs:
                claims_with_evidence_count += 1
                retrieved_ids = [p.passage_id for p in passages]
                retrieved_titles = [p.title.lower() for p in passages]

                # Check if any gold reference document is present in retrieved passages
                hit_rank = 0
                for r_idx, p in enumerate(passages, start=1):
                    if p.passage_id in evidence_refs or any(g.lower() in p.title.lower() or p.title.lower() in g.lower() for g in evidence_refs):
                        hit_rank = r_idx
                        break

                if hit_rank > 0:
                    retrieval_hits += 1
                    mrr_total += 1.0 / hit_rank

            detailed_results.append({
                "id": claim_id,
                "claim": claim_text,
                "gold_label": gold,
                "predicted_label": pred,
                "confidence": v_res.confidence,
                "passages_retrieved": len(passages),
                "is_correct": bool(gold == pred)
            })

        # Classification metrics
        acc = float(accuracy_score(gold_labels, predicted_labels))
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            gold_labels, predicted_labels, labels=VALID_LABELS, average="macro", zero_division=0
        )
        per_class_p, per_class_r, per_class_f1, per_class_sup = precision_recall_fscore_support(
            gold_labels, predicted_labels, labels=VALID_LABELS, zero_division=0
        )
        cm = confusion_matrix(gold_labels, predicted_labels, labels=VALID_LABELS).tolist()

        # Retrieval metrics
        recall_at_k = (retrieval_hits / claims_with_evidence_count) if claims_with_evidence_count > 0 else 0.0
        mrr = (mrr_total / claims_with_evidence_count) if claims_with_evidence_count > 0 else 0.0

        metrics = {
            "pipeline_name": "EvidenceLens Evidence-Grounded Fact Verification Pipeline",
            "eval_dataset": str(fever_path.name),
            "total_samples": len(records),
            "classification_metrics": {
                "accuracy": round(acc, 4),
                "macro_precision": round(float(macro_p), 4),
                "macro_recall": round(float(macro_r), 4),
                "macro_f1": round(float(macro_f1), 4),
                "per_class": {
                    lbl: {
                        "precision": round(float(per_class_p[i]), 4),
                        "recall": round(float(per_class_r[i]), 4),
                        "f1_score": round(float(per_class_f1[i]), 4),
                        "support": int(per_class_sup[i])
                    }
                    for i, lbl in enumerate(VALID_LABELS)
                },
                "confusion_matrix": {
                    "labels": VALID_LABELS,
                    "matrix": cm
                }
            },
            "retrieval_metrics": {
                "evaluated_claims_with_evidence": claims_with_evidence_count,
                "recall_at_top_k": round(recall_at_k, 4),
                "mrr": round(mrr, 4)
            },
            "detailed_predictions": detailed_results
        }
        return metrics
