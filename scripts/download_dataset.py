"""Automated Dataset Download Script for FEVER Benchmark.
Provides options to fetch official FEVER v1.0 splits from the official project repository
or verify existing local datasets.
"""

import os
import sys
import urllib.request
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR, SAMPLE_DATA_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("download_dataset")

# Official FEVER download URLs (maintained by fever.ai authors)
FEVER_FILES = {
    "train.jsonl": "https://fever.ai/download/fever/train.jsonl",
    "dev.jsonl": "https://fever.ai/download/fever/shared_task_dev.jsonl",
    "test.jsonl": "https://fever.ai/download/fever/shared_task_test.jsonl",
}

def download_file(url: str, dest_path: Path):
    """Downloads a file with basic progress logging and proper User-Agent header."""
    logger.info(f"Downloading from {url} to {dest_path}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
            totalsize = int(response.headers.get("Content-Length", 0))
            blocksize = 8192
            downloaded = 0
            while True:
                chunk = response.read(blocksize)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if totalsize > 0 and (downloaded // blocksize) % 100 == 0:
                    percent = min(100.0, downloaded * 100 / totalsize)
                    mb = downloaded / (1024 * 1024)
                    total_mb = totalsize / (1024 * 1024)
                    print(f"\rProgress: {percent:.1f}% ({mb:.1f} MB / {total_mb:.1f} MB)", end="", flush=True)
            print()
        logger.info(f"Successfully downloaded: {dest_path}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download FEVER official datasets")
    parser.add_argument("--splits", nargs="+", default=["dev.jsonl"],
                        choices=["train.jsonl", "dev.jsonl", "test.jsonl"],
                        help="Splits to download (dev.jsonl recommended for testing)")
    parser.add_argument("--dest", type=str, default=str(RAW_DATA_DIR),
                        help="Destination directory")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        url = FEVER_FILES.get(split)
        if url:
            target = dest_dir / split
            if target.exists() and target.stat().st_size > 0:
                logger.info(f"{split} already exists at {target}, skipping download.")
            else:
                download_file(url, target)

if __name__ == "__main__":
    main()
