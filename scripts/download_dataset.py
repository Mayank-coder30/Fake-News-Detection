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

# Official FEVER download URLs (S3 mirror maintained by FEVER authors)
FEVER_FILES = {
    "train.jsonl": "https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl",
    "dev.jsonl": "https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl",
    "test.jsonl": "https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl",
}

def download_file(url: str, dest_path: Path):
    """Downloads a file with basic progress logging."""
    logger.info(f"Downloading from {url} to {dest_path}...")
    try:
        def reporthook(blocknum, blocksize, totalsize):
            if totalsize > 0 and blocknum % 500 == 0:
                percent = min(100.0, blocknum * blocksize * 100 / totalsize)
                print(f"\rProgress: {percent:.1f}% ({blocknum * blocksize // (1024*1024)} MB)", end="")

        urllib.request.urlretrieve(url, str(dest_path), reporthook=reporthook)
        print()
        logger.info(f"Successfully downloaded: {dest_path}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")

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
            if target.exists():
                logger.info(f"{split} already exists at {target}, skipping download.")
            else:
                download_file(url, target)

if __name__ == "__main__":
    main()
