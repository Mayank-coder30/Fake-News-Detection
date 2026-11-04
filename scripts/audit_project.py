"""Comprehensive Project Health and Correctness Audit for EvidenceLens.
Validates file structures, data schemas, vector index alignment, model checkpoints,
pipeline components, and evaluation reports.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def audit():
    print("=" * 70)
    print("EVIDENCELENS FULL PROJECT AUDIT & SANITY REPORT")
    print("=" * 70)

    errors = 0
    warnings = 0

    # 1. Essential Project Files Check
    print("\n--- 1. ESSENTIAL CONFIGURATION & REPOSITORY FILES ---")
    essential_files = [
        "config.py",
        "app.py",
        "requirements.txt",
        "pytest.ini",
        ".env.example",
        "README.md",
        "docs/ARCHITECTURE.md",
        "docs/METHODOLOGY.md",
        "docs/DATASET.md",
        "docs/EVALUATION.md",
        "docs/LIMITATIONS.md",
        "docs/VIVA_PREPARATION.md"
    ]
    for rel_path in essential_files:
        p = Path(rel_path)
        if p.exists() and p.stat().st_size > 0:
            print(f"  [OK] {rel_path:<32} ({p.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  [FAIL] {rel_path:<32} MISSING OR EMPTY")
            errors += 1

    # 2. Data Files Integrity Check
    print("\n--- 2. DATASETS & JSONL SCHEMA VERIFICATION ---")
    data_specs = {
        "data/raw/dev.jsonl": {"min": 10000, "expected_fields": ["id", "claim", "label"]},
        "data/sample/fever_sample.jsonl": {"min": 500, "expected_fields": ["id", "claim", "label"]},
        "data/processed/fever_processed.jsonl": {"min": 500, "expected_fields": ["id", "claim", "label", "evidence_count"]},
        "data/sample/evidence_corpus.jsonl": {"min": 100, "expected_fields": ["passage_id", "title", "text"]},
        "data/sample/baseline_fake_news.jsonl": {"min": 100, "expected_fields": ["id", "text", "label"]},
        "data/index/evidence_metadata.jsonl": {"min": 100, "expected_fields": ["passage_id", "title"]}
    }

    for path_str, spec in data_specs.items():
        p = Path(path_str)
        if not p.exists():
            print(f"  [FAIL] {path_str:<38} NOT FOUND")
            errors += 1
            continue

        count = 0
        corrupt = 0
        missing_fields = 0
        with open(p, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                count += 1
                try:
                    obj = json.loads(line)
                    for req_f in spec["expected_fields"]:
                        if req_f not in obj:
                            missing_fields += 1
                except Exception:
                    corrupt += 1

        if corrupt == 0 and missing_fields == 0 and count >= spec["min"]:
            print(f"  [OK]   {path_str:<38} {count:>5} records (Valid JSONL, schemas match)")
        else:
            print(f"  [WARN] {path_str:<38} {count:>5} records (Corrupt: {corrupt}, Missing fields: {missing_fields})")
            if count < spec["min"]:
                warnings += 1

    # 3. FAISS Vector Index vs Metadata Alignment
    print("\n--- 3. FAISS VECTOR INDEX & EMBEDDING ALIGNMENT ---")
    faiss_path = Path("data/index/evidence_index.faiss")
    meta_path = Path("data/index/evidence_metadata.jsonl")
    if faiss_path.exists() and meta_path.exists():
        import faiss
        idx = faiss.read_index(str(faiss_path))
        meta_count = sum(1 for line in open(meta_path, "r", encoding="utf-8") if line.strip())
        if idx.ntotal == meta_count and idx.d == 384:
            print(f"  [OK]   FAISS Vector Index: {idx.ntotal} vectors, dimension={idx.d} (Exact match with metadata)")
        else:
            print(f"  [WARN] Index count ({idx.ntotal}) != metadata count ({meta_count}) or dim != 384")
            warnings += 1
    else:
        print(f"  [FAIL] FAISS index or metadata file missing on disk")
        errors += 1

    # 4. Model Checkpoint
    print("\n--- 4. SERIALIZED MODEL CHECKPOINTS ---")
    for mp_str in ["models/baseline_classifier.joblib", "models/fever_baseline_classifier.joblib"]:
        mp = Path(mp_str)
        if mp.exists() and mp.stat().st_size > 0:
            import joblib
            clf = joblib.load(str(mp))
            classes = list(getattr(clf, "classes_", []))
            print(f"  [OK]   Model: {mp.name:<32} ({mp.stat().st_size/1024:.1f} KB), classes={classes}")
        else:
            print(f"  [FAIL] Model checkpoint missing: {mp_str}")
            errors += 1

    # 5. Pipeline End-to-End Live Smoke Test
    print("\n--- 5. PIPELINE EXECUTION & VERIFICATION TEST ---")
    try:
        from src.pipeline.fact_check_pipeline import FactCheckPipeline
        pipeline = FactCheckPipeline()
        test_article = (
            "The James Webb Space Telescope conducts infrared astronomy. "
            "Conspiracy rumors claim that Apollo 11 actually landed on Mars in 1969. "
            "Albert Einstein owned an orange tabby cat named Copernicus."
        )
        out = pipeline.analyze_article(test_article)
        print(f"  [OK]   FactCheckPipeline instantiated successfully")
        print(f"  [OK]   Analyzed test copy ({len(out.claims)} claims extracted):")
        for clm in out.claims:
            verdict = clm["label"]
            conf = clm.get("confidence", 0.0) * 100
            print(f"         - \"{clm['claim_text'][:52]}...\" -> {verdict} ({conf:.1f}%)")
    except Exception as e:
        print(f"  [FAIL] Pipeline test failed with error: {e}")
        errors += 1

    # 6. Streamlit UI Health Check
    print("\n--- 6. STREAMLIT APPLICATION VALIDATION ---")
    try:
        import app
        print(f"  [OK]   Streamlit application (app.py) imports without syntax or import errors")
    except Exception as e:
        print(f"  [FAIL] app.py import failed: {e}")
        errors += 1

    # 7. Metrics & Reports Verification
    print("\n--- 7. EVALUATION REPORTS & BENCHMARK METRICS ---")
    report_files = [
        "reports/baseline_metrics.json",
        "reports/fever_baseline_metrics.json",
        "reports/pipeline_metrics.json",
        "reports/model_comparison.md"
    ]
    for rf in report_files:
        p = Path(rf)
        if p.exists() and p.stat().st_size > 0:
            print(f"  [OK]   {rf:<34} ({p.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  [FAIL] {rf:<34} MISSING")
            errors += 1

    # Final summary
    print("\n" + "=" * 70)
    if errors == 0 and warnings == 0:
        print("AUDIT RESULT: ALL SYSTEMS OPERATIONAL (0 ERRORS, 0 WARNINGS)")
    else:
        print(f"AUDIT RESULT: {errors} ERROR(S), {warnings} WARNING(S)")
    print("=" * 70)

if __name__ == "__main__":
    audit()

