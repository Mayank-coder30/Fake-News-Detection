"""
Fetches Wikipedia passages for a specific list of article titles (from a file),
merges them into the existing evidence corpus, and rebuilds the FAISS index.

Usage:
    python scripts/fetch_eval_evidence.py --titles-file /tmp/missing_titles.txt
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DEFAULT_EVIDENCE_CORPUS,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
)
from src.utils.logging_utils import get_logger

logger = get_logger("fetch_eval_evidence")

WIKI_API = "https://en.wikipedia.org/w/api.php"
SECTION_LIMIT = 3        # paragraphs per article
MIN_CHARS = 80
REQUEST_DELAY = 0.15     # seconds between API calls
HEADERS = {
    "User-Agent": "EvidenceLens/1.0 (NIT Delhi Academic Project; rachitsinghrana@example.com) python-requests/2.31"
}


# ── FEVER title decoding ────────────────────────────────────────────────────
_FEVER_MAP = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-COLON-": ":",
    "-SLASH-": "/",
    "-LSB-": "[",
    "-RSB-": "]",
}

def decode_fever_title(title: str) -> str:
    for token, char in _FEVER_MAP.items():
        title = title.replace(token, char)
    return title.replace("_", " ")


# ── Wikipedia fetcher ────────────────────────────────────────────────────────
def fetch_passages(title: str) -> list[dict]:
    display = decode_fever_title(title)
    params = {
        "action": "query",
        "titles": display,
        "prop": "extracts",
        "exintro": False,
        "explaintext": True,
        "exsectionformat": "plain",
        "redirects": 1,
        "format": "json",
    }
    try:
        resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
    except Exception as e:
        logger.warning(f"  API error for '{display}': {e}")
        return []

    passages = []
    for page_id, page in pages.items():
        if page_id == "-1" or "missing" in page:
            continue
        text = page.get("extract", "") or ""
        # Split on double-newline paragraphs
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        for idx, para in enumerate(paras[:SECTION_LIMIT]):
            if len(para) < MIN_CHARS:
                continue
            pid = f"{title}_{idx}"
            passages.append({
                "passage_id": pid,
                "id": pid,
                "title": display,
                "text": para,
                "source_id": f"wiki_{title[:30]}",
                "source_url": f"https://en.wikipedia.org/wiki/{display.replace(' ', '_')}",
            })
    return passages


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--titles-file", required=True, help="One FEVER title per line")
    args = parser.parse_args()

    titles_path = Path(args.titles_file)
    if not titles_path.exists():
        logger.error(f"Titles file not found: {titles_path}")
        sys.exit(1)

    titles = [t.strip() for t in titles_path.read_text().splitlines() if t.strip()]
    logger.info(f"Titles to fetch: {len(titles)}")

    # Load existing corpus to avoid duplicates
    existing_ids: set[str] = set()
    existing_passages: list[dict] = []
    if DEFAULT_EVIDENCE_CORPUS.exists():
        with open(DEFAULT_EVIDENCE_CORPUS, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    p = json.loads(line)
                    existing_ids.add(p.get("passage_id") or p.get("id", ""))
                    existing_passages.append(p)
    logger.info(f"Existing corpus size: {len(existing_passages)}")

    # Fetch new passages
    new_passages = []
    for i, title in enumerate(titles, start=1):
        passages = fetch_passages(title)
        for p in passages:
            if p["passage_id"] not in existing_ids:
                new_passages.append(p)
                existing_ids.add(p["passage_id"])
        if i % 50 == 0 or i == len(titles):
            logger.info(f"  [{i}/{len(titles)}] fetched — new passages so far: {len(new_passages)}")
        time.sleep(REQUEST_DELAY)

    logger.info(f"New passages fetched: {len(new_passages)}")

    # Merge and write corpus — normalise all records to have an "id" field
    all_passages = []
    for p in existing_passages:
        if "id" not in p:
            p["id"] = p.get("passage_id", "")
        all_passages.append(p)
    all_passages.extend(new_passages)
    with open(DEFAULT_EVIDENCE_CORPUS, "w", encoding="utf-8") as f:
        for p in all_passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info(f"Corpus updated: {len(all_passages)} total passages → {DEFAULT_EVIDENCE_CORPUS}")

    # Rebuild FAISS index
    logger.info("Rebuilding FAISS index...")
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        texts = [p["text"] for p in all_passages]
        logger.info(f"  Encoding {len(texts)} passages...")
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        logger.info(f"  FAISS index saved: {index.ntotal} vectors → {FAISS_INDEX_PATH}")

        with open(FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
            for p in all_passages:
                pid = p.get("passage_id") or p.get("id", "")
                f.write(json.dumps({"id": pid, "title": p["title"]}, ensure_ascii=False) + "\n")
        logger.info(f"  Metadata saved → {FAISS_METADATA_PATH}")

    except ImportError:
        logger.error("faiss not found in this environment — index rebuild skipped.")
        sys.exit(1)

    logger.info("Done. Run run_pipeline_eval_sampled.py to re-evaluate.")
    print(f"\nCorpus: {len(all_passages)} passages | FAISS: {index.ntotal} vectors")


if __name__ == "__main__":
    main()
