"""Fetch Authentic Wikipedia Evidence Passages for FEVER Dataset.

Connects to the official Wikimedia API to batch-download Wikipedia introductory
extracts for the most frequently referenced entities in FEVER.
Expands the evidence corpus and rebuilds the FAISS vector index.
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Set
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_EVIDENCE_CORPUS, PROCESSED_DATA_DIR, FAISS_INDEX_PATH, FAISS_METADATA_PATH
from src.utils.logging_utils import get_logger
from scripts.build_evidence_index import build_index

logger = get_logger("fetch_wikipedia_evidence")

MEDIAWIKI_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "EvidenceLensResearch/1.0 (B.Tech CSE Project at NIT Delhi; contact: 241210097@nitdelhi.ac.in)"


def decode_fever_title(raw_title: str) -> str:
    """Decodes FEVER-specific title encodings into standard Wikipedia title strings."""
    return (
        raw_title.replace("-LRB-", "(")
        .replace("-RRB-", ")")
        .replace("-COLON-", ":")
        .replace("-SLASH-", "/")
        .replace("_", " ")
        .strip()
    )


def get_top_fever_titles(fever_path: Path, max_titles: int = 500) -> List[str]:
    """Finds the most frequently referenced Wikipedia document titles in FEVER."""
    if not fever_path.exists():
        logger.error(f"FEVER dataset missing at {fever_path}")
        return []

    logger.info(f"Extracting top {max_titles} referenced titles from {fever_path}...")
    counts = Counter()
    with open(fever_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            for ref in rec.get("evidence_refs", []):
                t = ref.get("doc_title")
                if t:
                    counts[t] += 1

    top_raw = [t for t, _ in counts.most_common(max_titles)]
    logger.info(f"Selected top {len(top_raw)} most referenced Wikipedia articles (covering {sum(counts[t] for t in top_raw)} evidence citations).")
    return top_raw


def batch_fetch_wikipedia_extracts(titles: List[str], batch_size: int = 40) -> Dict[str, str]:
    """Fetches Wikipedia article introductory extracts in batches via MediaWiki action API."""
    results = {}
    total_batches = (len(titles) + batch_size - 1) // batch_size

    for b_idx in range(total_batches):
        batch = titles[b_idx * batch_size : (b_idx + 1) * batch_size]
        title_str = "|".join(batch)

        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "titles": title_str,
            "redirects": "1"
        }

        url = f"{MEDIAWIKI_API_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})

                for pid, pdata in pages.items():
                    if pid == "-1" or "missing" in pdata:
                        continue
                    t = pdata.get("title", "")
                    extract = pdata.get("extract", "").strip()
                    if t and extract and len(extract) >= 40:
                        # Normalize whitespace
                        clean_text = " ".join(extract.split())
                        # Store by normalized title
                        results[t] = clean_text

            logger.info(f"Batch {b_idx + 1}/{total_batches}: Fetched {len(results)} total articles so far.")
            time.sleep(0.3)  # Gentle rate limit respect
        except Exception as e:
            logger.warning(f"Batch {b_idx + 1} request error: {e}")
            time.sleep(1.0)

    return results


def expand_corpus(target_total: int = 500):
    """Main routine to expand evidence corpus with authentic Wikipedia pages and rebuild FAISS index."""
    # 1. Load existing corpus to preserve original curated passages
    existing_passages = []
    existing_titles: Set[str] = set()

    if DEFAULT_EVIDENCE_CORPUS.exists():
        with open(DEFAULT_EVIDENCE_CORPUS, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing_passages.append(item)
                    existing_titles.add(item.get("title", "").lower())
        logger.info(f"Loaded {len(existing_passages)} existing passages from {DEFAULT_EVIDENCE_CORPUS}.")

    # 2. Extract top referenced FEVER titles
    fever_file = PROCESSED_DATA_DIR / "fever_processed.jsonl"
    raw_titles = get_top_fever_titles(fever_file, max_titles=target_total)

    # Map raw title to decoded Wikipedia title
    title_map = {decode_fever_title(rt): rt for rt in raw_titles}
    wiki_titles = list(title_map.keys())

    # 3. Fetch from Wikipedia
    extracts = batch_fetch_wikipedia_extracts(wiki_titles, batch_size=40)
    logger.info(f"Successfully retrieved {len(extracts)} genuine Wikipedia articles.")

    # 4. Integrate into corpus
    new_docs_count = 0
    next_doc_id = len(existing_passages) + 1

    for title, extract_text in extracts.items():
        if title.lower() in existing_titles:
            continue

        raw_key = title_map.get(title, title.replace(" ", "_"))
        # Truncate text to standard paragraph length (first ~1000 chars or 3-4 sentences)
        sentences = extract_text.split(". ")
        summary_text = ". ".join(sentences[:5])
        if not summary_text.endswith("."):
            summary_text += "."

        doc_entry = {
            "passage_id": f"doc_{next_doc_id:03d}",
            "title": title,
            "text": summary_text,
            "source_id": f"wiki_{raw_key.lower()}",
            "source_url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        }
        existing_passages.append(doc_entry)
        existing_titles.add(title.lower())
        next_doc_id += 1
        new_docs_count += 1

    # 5. Persist updated corpus
    logger.info(f"Writing {len(existing_passages)} total passages ({new_docs_count} new) to {DEFAULT_EVIDENCE_CORPUS}...")
    DEFAULT_EVIDENCE_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_EVIDENCE_CORPUS, "w", encoding="utf-8") as f:
        for p in existing_passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info(f"Corpus updated: {len(existing_passages)} total passages saved.")

    # 6. Rebuild FAISS vector index
    logger.info("Rebuilding FAISS vector index with expanded evidence corpus...")
    build_index(
        corpus_path=DEFAULT_EVIDENCE_CORPUS,
        index_out=FAISS_INDEX_PATH,
        meta_out=FAISS_METADATA_PATH
    )
    logger.info("FAISS evidence index successfully re-indexed and synchronized!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch Wikipedia evidence for FEVER dataset")
    parser.add_argument("--count", type=int, default=500, help="Number of top Wikipedia articles to fetch")
    args = parser.parse_args()

    expand_corpus(target_total=args.count)
