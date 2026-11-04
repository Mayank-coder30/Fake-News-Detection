"""
Runs EvidenceLens pipeline evaluation on a stratified random sample of FEVER claims.

Samples N claims per class (SUPPORTED / REFUTED / NOT ENOUGH INFO) from the
processed FEVER dataset, evaluates each through FAISS retrieval + NLI verification,
and writes the updated pipeline_metrics.json.

Usage:
    python scripts/run_pipeline_eval_sampled.py [--per-class N]

Default: 334 per class → ~1000 total claims evaluated.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    LABEL_SUPPORTED,
    LABEL_REFUTED,
    LABEL_UNCERTAIN,
    VALID_LABELS,
)
from src.pipeline.fact_check_pipeline import FactCheckPipeline
from src.utils.logging_utils import get_logger

logger = get_logger("run_pipeline_eval_sampled")

SEED = 42


def load_stratified_sample(fever_path: Path, per_class: int) -> list:
    """Return up to `per_class` records per label, randomly sampled."""
    buckets: dict[str, list] = {lbl: [] for lbl in VALID_LABELS}

    with open(fever_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lbl = rec.get("label", "")
            if lbl in buckets:
                buckets[lbl].append(rec)

    rng = random.Random(SEED)
    sample = []
    for lbl, recs in buckets.items():
        chosen = rng.sample(recs, min(per_class, len(recs)))
        logger.info(f"  {lbl}: sampled {len(chosen)} / {len(recs)} total")
        sample.extend(chosen)

    rng.shuffle(sample)
    return sample


def evaluate(records: list, pipeline: FactCheckPipeline) -> dict:
    """Run the pipeline over records and return metrics dict."""
    gold_labels, predicted_labels = [], []
    retrieval_hits = 0
    mrr_total = 0.0
    claims_with_evidence_count = 0
    detailed_results = []

    total = len(records)
    for idx, rec in enumerate(records, start=1):
        claim_id = str(rec["id"])
        claim_text = rec["claim"]
        gold = rec["label"]
        evidence_refs = [ref["doc_title"] for ref in rec.get("evidence_refs", [])]

        passages = pipeline.retriever.retrieve_for_claim(claim_text, top_k=pipeline.top_k)
        v_res = pipeline.verifier.verify(claim_id, claim_text, passages)

        pred = v_res.label
        gold_labels.append(gold)
        predicted_labels.append(pred)

        if evidence_refs:
            claims_with_evidence_count += 1
            hit_rank = 0
            for r_idx, p in enumerate(passages, start=1):
                if p.passage_id in evidence_refs or any(
                    g.lower() in p.title.lower() or p.title.lower() in g.lower()
                    for g in evidence_refs
                ):
                    hit_rank = r_idx
                    break
            if hit_rank > 0:
                retrieval_hits += 1
                mrr_total += 1.0 / hit_rank

        detailed_results.append(
            {
                "id": claim_id,
                "claim": claim_text,
                "gold_label": gold,
                "predicted_label": pred,
                "confidence": v_res.confidence,
                "passages_retrieved": len(passages),
                "is_correct": bool(gold == pred),
            }
        )

        if idx % 100 == 0 or idx == total:
            correct_so_far = sum(1 for r in detailed_results if r["is_correct"])
            logger.info(
                f"  [{idx}/{total}] running accuracy: {correct_so_far/idx*100:.1f}%"
            )

    # ---- Metrics ----
    acc = float(accuracy_score(gold_labels, predicted_labels))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        gold_labels, predicted_labels, labels=VALID_LABELS, average="macro", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, per_class_sup = precision_recall_fscore_support(
        gold_labels, predicted_labels, labels=VALID_LABELS, zero_division=0
    )
    cm = confusion_matrix(gold_labels, predicted_labels, labels=VALID_LABELS).tolist()

    recall_at_k = (retrieval_hits / claims_with_evidence_count) if claims_with_evidence_count > 0 else 0.0
    mrr = (mrr_total / claims_with_evidence_count) if claims_with_evidence_count > 0 else 0.0

    return {
        "pipeline_name": "EvidenceLens Evidence-Grounded Fact Verification Pipeline",
        "eval_dataset": f"FEVER Stratified Sample (seed={SEED})",
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
                    "support": int(per_class_sup[i]),
                }
                for i, lbl in enumerate(VALID_LABELS)
            },
            "confusion_matrix": {"labels": VALID_LABELS, "matrix": cm},
        },
        "retrieval_metrics": {
            "recall_at_top_k": round(recall_at_k, 4),
            "mrr": round(mrr, 4),
            "top_k": pipeline.top_k,
            "relevance_threshold": pipeline.retriever.relevance_threshold,
            "claims_with_evidence": claims_with_evidence_count,
            "retrieval_hits": retrieval_hits,
        },
        "detailed_predictions": detailed_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Stratified FEVER pipeline evaluation")
    parser.add_argument(
        "--per-class",
        type=int,
        default=334,
        help="Number of claims to sample per class (default: 334 → ~1002 total)",
    )
    args = parser.parse_args()

    fever_path = PROCESSED_DATA_DIR / "fever_processed.jsonl"
    if not fever_path.exists():
        logger.error(f"Processed FEVER file not found: {fever_path}")
        sys.exit(1)

    logger.info(f"Loading stratified sample ({args.per_class} per class) from {fever_path}")
    records = load_stratified_sample(fever_path, args.per_class)
    logger.info(f"Total claims to evaluate: {len(records)}")

    logger.info("Initialising FactCheckPipeline...")
    pipeline = FactCheckPipeline()

    logger.info("Running pipeline evaluation...")
    metrics = evaluate(records, pipeline)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "pipeline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics written to: {out_path}")

    cls_m = metrics["classification_metrics"]
    ret_m = metrics["retrieval_metrics"]
    print("\n" + "=" * 60)
    print("EVIDENCELENS PIPELINE — STRATIFIED FEVER EVALUATION")
    print("=" * 60)
    print(f"Total Evaluated Claims : {metrics['total_samples']}")
    print(f"Overall Accuracy       : {cls_m['accuracy']:.4f}  ({cls_m['accuracy']*100:.1f}%)")
    print(f"Macro F1-Score         : {cls_m['macro_f1']:.4f}")
    print(f"Retrieval Recall@k     : {ret_m['recall_at_top_k']:.4f}")
    print(f"Retrieval MRR          : {ret_m['mrr']:.4f}")
    print("\nPer-Class Performance:")
    for lbl, stats in cls_m["per_class"].items():
        print(
            f"  {lbl:<12} | P: {stats['precision']:.4f}  R: {stats['recall']:.4f}"
            f"  F1: {stats['f1_score']:.4f}  (n={stats['support']})"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
