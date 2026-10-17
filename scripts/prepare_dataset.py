"""Dataset Preparation and Standardization Script for EvidenceLens.
Loads raw FEVER JSONL data, maps labels (SUPPORTS -> SUPPORTED, REFUTES -> REFUTED,
NOT ENOUGH INFO -> UNCERTAIN), validates evidence annotations, and exports clean splits.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_DATA_DIR,
    LABEL_SUPPORTED,
    LABEL_REFUTED,
    LABEL_UNCERTAIN
)
from src.utils.logging_utils import get_logger

logger = get_logger("prepare_dataset")

# Canonical label mapping from FEVER official taxonomy
FEVER_LABEL_MAPPING = {
    "SUPPORTS": LABEL_SUPPORTED,
    "REFUTES": LABEL_REFUTED,
    "NOT ENOUGH INFO": LABEL_UNCERTAIN
}

def clean_and_standardize_record(raw_record: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and maps a single raw FEVER record into standard schema."""
    claim_id = raw_record.get("id")
    claim_text = raw_record.get("claim", "").strip()
    raw_label = raw_record.get("label", "").strip()

    if not claim_id or not claim_text or not raw_label:
        return None

    mapped_label = FEVER_LABEL_MAPPING.get(raw_label.upper())
    if not mapped_label:
        return None

    # Process evidence references
    evidence_entries = []
    raw_evidence = raw_record.get("evidence", [])
    if isinstance(raw_evidence, list):
        for ev in raw_evidence:
            if isinstance(ev, list) and len(ev) > 0:
                # In FEVER format: [annotation_id, evidence_id, doc_title, line_number]
                doc_title = ev[2] if len(ev) > 2 and ev[2] is not None else None
                line_no = ev[3] if len(ev) > 3 and ev[3] is not None else None
                if doc_title:
                    evidence_entries.append({
                        "doc_title": str(doc_title),
                        "line_number": line_no
                    })

    return {
        "id": claim_id,
        "claim": claim_text,
        "label": mapped_label,
        "original_label": raw_label,
        "evidence_count": len(evidence_entries),
        "evidence_refs": evidence_entries
    }

def prepare_dataset(input_file: Path, output_file: Path) -> Dict[str, int]:
    """Reads input JSONL file, standardizes records, and writes to processed directory."""
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return {}

    logger.info(f"Processing dataset from: {input_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_records": 0,
        "valid_records": 0,
        "malformed_skipped": 0,
        LABEL_SUPPORTED: 0,
        LABEL_REFUTED: 0,
        LABEL_UNCERTAIN: 0
    }

    processed_records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            stats["total_records"] += 1
            try:
                raw_rec = json.loads(line)
                std_rec = clean_and_standardize_record(raw_rec)
                if std_rec is None:
                    stats["malformed_skipped"] += 1
                    continue
                stats["valid_records"] += 1
                stats[std_rec["label"]] += 1
                processed_records.append(std_rec)
            except Exception as e:
                stats["malformed_skipped"] += 1
                logger.warning(f"Error parsing line {line_num}: {e}")

    with open(output_file, "w", encoding="utf-8") as out:
        for rec in processed_records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(processed_records)} standardized records to: {output_file}")
    logger.info(f"Dataset Summary: {stats}")
    return stats

if __name__ == "__main__":
    # Prepare from sample or raw if available
    sample_fever = SAMPLE_DATA_DIR / "fever_sample.jsonl"
    processed_out = PROCESSED_DATA_DIR / "fever_processed.jsonl"
    prepare_dataset(sample_fever, processed_out)
