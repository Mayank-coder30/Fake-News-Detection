"""Master Evaluation Script for EvidenceLens.
Executes evaluation on both the conventional baseline and proposed pipeline,
computing accuracy, macro-F1, per-class metrics, retrieval recall, and generating comparative reports.
"""

import json
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR, REPORTS_DIR
from src.pipeline.fact_check_pipeline import FactCheckPipeline
from src.evaluation.evaluate_verifier import VerifierEvaluator
from src.evaluation.compare_models import generate_comparative_report
from src.utils.logging_utils import get_logger

logger = get_logger("run_evaluation")

def run_all_evaluations():
    fever_eval_file = PROCESSED_DATA_DIR / "fever_processed.jsonl"
    if not fever_eval_file.exists():
        logger.error(f"Processed FEVER file missing at {fever_eval_file}. Run scripts/prepare_dataset.py first.")
        return

    logger.info("Initializing FactCheckPipeline for benchmark evaluation...")
    pipeline = FactCheckPipeline()

    evaluator = VerifierEvaluator(pipeline)
    pipeline_metrics = evaluator.evaluate_on_fever(fever_eval_file)

    # Save pipeline metrics
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pipe_out = REPORTS_DIR / "pipeline_metrics.json"
    with open(pipe_out, "w", encoding="utf-8") as f:
        json.dump(pipeline_metrics, f, indent=2)
    logger.info(f"Pipeline metrics written to: {pipe_out}")

    # Generate comparative markdown report
    comp_result = generate_comparative_report()
    logger.info(f"Generated comparison report: {comp_result['report_path']}")

    # Print summary to console
    cls_m = pipeline_metrics["classification_metrics"]
    ret_m = pipeline_metrics["retrieval_metrics"]
    print("\n" + "="*60)
    print("EVIDENCELENS FACT VERIFICATION EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Evaluated Claims: {pipeline_metrics['total_samples']}")
    print(f"Overall Accuracy:       {cls_m['accuracy']:.4f}")
    print(f"Macro F1-Score:         {cls_m['macro_f1']:.4f}")
    print(f"Retrieval Recall@k:     {ret_m['recall_at_top_k']:.4f}")
    print(f"Retrieval MRR:          {ret_m['mrr']:.4f}")
    print("\nPer-Class Performance:")
    for lbl, stats in cls_m["per_class"].items():
        print(f"  {lbl:<12} | Precision: {stats['precision']:.4f} | Recall: {stats['recall']:.4f} | F1: {stats['f1_score']:.4f} (Support: {stats['support']})")
    print("="*60)

if __name__ == "__main__":
    run_all_evaluations()
