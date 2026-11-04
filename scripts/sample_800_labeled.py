"""Sample N = 800 labeled rows from dev.jsonl and prepare processed FEVER dataset."""
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR, SAMPLE_DATA_DIR, PROCESSED_DATA_DIR
from scripts.prepare_dataset import clean_and_standardize_record

raw_file = RAW_DATA_DIR / "dev.jsonl"
sample_file = SAMPLE_DATA_DIR / "fever_sample.jsonl"
processed_file = PROCESSED_DATA_DIR / "fever_processed.jsonl"

print(f"Reading raw data from {raw_file}...")
buckets = {
    "SUPPORTS": [],
    "REFUTES": [],
    "NOT ENOUGH INFO": []
}

with open(raw_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            lbl = rec.get("label", "").strip()
            # Ensure the row has valid labeled data
            if lbl in buckets and rec.get("claim"):
                buckets[lbl].append(rec)
        except json.JSONDecodeError:
            continue

print(f"Available labeled records:")
for k, v in buckets.items():
    print(f"  {k}: {len(v)}")

# Sample exactly 267 SUPPORTS, 267 REFUTES, 266 NOT ENOUGH INFO = 800 total
random.seed(42)
sampled = []
sampled.extend(random.sample(buckets["SUPPORTS"], 267))
sampled.extend(random.sample(buckets["REFUTES"], 267))
sampled.extend(random.sample(buckets["NOT ENOUGH INFO"], 266))

random.shuffle(sampled)

# Assign sequential clean IDs
for i, rec in enumerate(sampled, start=1001):
    rec["id"] = i

# Save to fever_sample.jsonl
sample_file.parent.mkdir(parents=True, exist_ok=True)
with open(sample_file, "w", encoding="utf-8") as f:
    for rec in sampled:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Saved {len(sampled)} records to {sample_file}")

# Standardize to fever_processed.jsonl
processed_records = []
for rec in sampled:
    std = clean_and_standardize_record(rec)
    if std is not None:
        processed_records.append(std)

with open(processed_file, "w", encoding="utf-8") as f:
    for rec in processed_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Saved {len(processed_records)} standardized records to {processed_file}")

# Verify counts by label
from collections import Counter
counts = Counter(r["label"] for r in processed_records)
print(f"Processed label distribution: {dict(counts)}")

