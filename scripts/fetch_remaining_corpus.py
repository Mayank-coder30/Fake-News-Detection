import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import time
import urllib.request
import urllib.parse
from config import DEFAULT_EVIDENCE_CORPUS, PROCESSED_DATA_DIR, FAISS_INDEX_PATH, FAISS_METADATA_PATH, DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIM
from scripts.expand_corpus_for_dataset import clean_fever_title
from scripts.build_evidence_index import build_index
from src.utils.logging_utils import get_logger

logger = get_logger("fetch_remaining")

def batch_fetch_wikipedia_with_retry(titles_chunk, max_retries=3):
    if not titles_chunk:
        return {}
    
    # URL encode titles properly; note: pipe '|' is the separator for MediaWiki
    encoded_titles = "|".join([urllib.parse.quote(t.replace(" ", "_")) for t in titles_chunk])
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&redirects=1&titles={encoded_titles}&format=json"
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "EvidenceLensAcademicProject/1.0 (contact: student@institution.edu; FEVER benchmark study)"}
    )
    
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                results = {}
                for pid, page in pages.items():
                    if pid == "-1" or "missing" in page:
                        continue
                    title = page.get("title")
                    extract = page.get("extract", "").strip()
                    if title and extract and len(extract) > 20:
                        results[title] = extract
                return results
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2 * (attempt + 1))
    return {}

def main():
    with open(DEFAULT_EVIDENCE_CORPUS, "r", encoding="utf-8") as f:
        existing_docs = [json.loads(line) for line in f]
    
    existing_titles = set()
    for d in existing_docs:
        existing_titles.add(d["title"].lower())
        existing_titles.add(d["title"].lower().replace(" ", "_"))
        existing_titles.add(d["title"].lower().replace("_", " "))

    with open(PROCESSED_DATA_DIR / "fever_processed.jsonl", "r", encoding="utf-8") as f:
        claims = [json.loads(line) for line in f]

    needed_raw = set()
    for c in claims:
        for ev in c.get("evidence_refs", []):
            rt = ev.get("doc_title")
            if rt:
                needed_raw.add(rt)

    to_fetch = []
    for raw in needed_raw:
        clean = clean_fever_title(raw)
        if clean.lower() not in existing_titles and clean.lower().replace(" ", "_") not in existing_titles:
            to_fetch.append(clean)

    logger.info(f"Existing corpus size: {len(existing_docs)}")
    logger.info(f"Unique cited titles still missing: {len(to_fetch)}")

    batch_size = 25
    new_docs = []
    curr_id = len(existing_docs) + 1

    for i in range(0, len(to_fetch), batch_size):
        chunk = to_fetch[i:i + batch_size]
        logger.info(f"Fetching batch {i // batch_size + 1} / {(len(to_fetch) + batch_size - 1) // batch_size} ({len(chunk)} titles)...")
        results = batch_fetch_wikipedia_with_retry(chunk)
        
        for title, extract in results.items():
            doc_id = f"doc_{curr_id:04d}"
            curr_id += 1
            slug = urllib.parse.quote(title.replace(" ", "_"))
            new_docs.append({
                "passage_id": doc_id,
                "title": title,
                "text": extract,
                "source_id": f"wiki_{slug.lower()}",
                "source_url": f"https://en.wikipedia.org/wiki/{slug}"
            })
            existing_titles.add(title.lower())
        
        time.sleep(1.0)

    logger.info(f"Successfully retrieved {len(new_docs)} additional Wikipedia passages!")
    all_docs = existing_docs + new_docs
    with open(DEFAULT_EVIDENCE_CORPUS, "w", encoding="utf-8") as f:
        for d in all_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    logger.info(f"Total corpus size now: {len(all_docs)}. Rebuilding FAISS index...")
    build_index(DEFAULT_EVIDENCE_CORPUS, FAISS_INDEX_PATH, FAISS_METADATA_PATH, DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIM)
    logger.info("Rebuild complete!")

if __name__ == "__main__":
    main()

