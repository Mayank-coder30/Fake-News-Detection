"""Expand Evidence Knowledge Base with Wikipedia articles cited in the dataset.
Fetches authentic Wikipedia summaries for cited entities and rebuilds FAISS index.
"""

import sys
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DEFAULT_EVIDENCE_CORPUS,
    PROCESSED_DATA_DIR,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIM
)
from scripts.build_evidence_index import build_index
from src.utils.logging_utils import get_logger

logger = get_logger("expand_corpus")

def clean_fever_title(title: str) -> str:
    """Decodes FEVER-specific title formatting to canonical Wikipedia titles."""
    t = title.replace("-LRB-", "(").replace("-RRB-", ")").replace("-COLON-", ":")
    t = t.replace("_", " ").strip()
    return t

def batch_fetch_wikipedia(titles_chunk):
    """Batch-fetches introductory extracts for a list of Wikipedia titles."""
    if not titles_chunk:
        return {}
    
    pipe_titles = "|".join([urllib.parse.quote(t.replace(" ", "_")) for t in titles_chunk])
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles={pipe_titles}&format=json"
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "EvidenceLensAcademicResearch/1.0 (BTech CS Project; contact: nitdelhi.ac.in)"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            results = {}
            for pid, page in pages.items():
                if pid == "-1" or "missing" in page:
                    continue
                title = page.get("title")
                extract = page.get("extract", "").strip()
                if title and extract and len(extract) > 20:
                    results[title.lower()] = {
                        "title": title,
                        "extract": extract
                    }
            return results
    except Exception as e:
        logger.warning(f"Batch fetch error: {e}")
        return {}

def main():
    processed_file = PROCESSED_DATA_DIR / "fever_processed.jsonl"
    corpus_file = DEFAULT_EVIDENCE_CORPUS

    logger.info("Loading existing evidence corpus...")
    existing_docs = []
    existing_titles = set()
    if corpus_file.exists():
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    existing_docs.append(doc)
                    existing_titles.add(doc["title"].lower())
                    existing_titles.add(doc["title"].lower().replace(" ", "_"))

    logger.info(f"Existing corpus passages: {len(existing_docs)}")

    # Extract all cited document titles from processed dataset
    logger.info(f"Extracting cited evidence titles from {processed_file}...")
    needed_titles = set()
    with open(processed_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                for ev in rec.get("evidence_refs", []):
                    raw_title = ev.get("doc_title")
                    if raw_title:
                        needed_titles.add(raw_title)

    logger.info(f"Total unique documents cited in dataset: {len(needed_titles)}")

    # Filter out already existing titles
    to_fetch = []
    for raw in needed_titles:
        clean = clean_fever_title(raw)
        if clean.lower() not in existing_titles and clean.lower().replace(" ", "_") not in existing_titles:
            to_fetch.append(clean)

    logger.info(f"New titles to fetch from Wikipedia API: {len(to_fetch)}")

    # Fetch in batches of 40
    batch_size = 40
    new_docs = []
    curr_id = len(existing_docs) + 1

    for i in range(0, len(to_fetch), batch_size):
        chunk = to_fetch[i:i + batch_size]
        logger.info(f"Fetching batch {i // batch_size + 1} / {(len(to_fetch) + batch_size - 1) // batch_size} ({len(chunk)} titles)...")
        results = batch_fetch_wikipedia(chunk)
        
        for k, v in results.items():
            doc_id = f"doc_{curr_id:04d}"
            curr_id += 1
            slug = urllib.parse.quote(v["title"].replace(" ", "_"))
            new_docs.append({
                "passage_id": doc_id,
                "title": v["title"],
                "text": v["extract"],
                "source_id": f"wiki_{slug.lower()}",
                "source_url": f"https://en.wikipedia.org/wiki/{slug}"
            })
            existing_titles.add(v["title"].lower())
        
        time.sleep(0.5)  # Respect API rate limits

    logger.info(f"Successfully retrieved {len(new_docs)} new Wikipedia passages!")

    # Merge and save expanded corpus
    all_docs = existing_docs + new_docs
    with open(corpus_file, "w", encoding="utf-8") as f:
        for d in all_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    logger.info(f"Expanded corpus saved to {corpus_file}. Total passages: {len(all_docs)}")

    # Rebuild FAISS index
    logger.info("Rebuilding FAISS dense vector index...")
    build_index(corpus_file, FAISS_INDEX_PATH, FAISS_METADATA_PATH, DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIM)
    logger.info("Index rebuild completed successfully!")

if __name__ == "__main__":
    main()

