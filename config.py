import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"

# Ensure essential runtime directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, MODELS_DIR, REPORTS_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset paths
DEFAULT_EVIDENCE_CORPUS = SAMPLE_DATA_DIR / "evidence_corpus.jsonl"
DEFAULT_FEVER_SAMPLE = SAMPLE_DATA_DIR / "fever_sample.jsonl"
BASELINE_DATASET_PATH = SAMPLE_DATA_DIR / "baseline_fake_news.jsonl"

# FAISS Vector Index paths
INDEX_DIR = DATA_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_PATH = INDEX_DIR / "evidence_index.faiss"
FAISS_METADATA_PATH = INDEX_DIR / "evidence_metadata.jsonl"

# Models Configuration
# Efficient, high-accuracy SentenceTransformer for evidence retrieval
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Pretrained Transformer NLI model for claim-evidence verification
# cross-encoder/nli-distilroberta-base maps claim-evidence pairs directly
DEFAULT_VERIFIER_MODEL = "cross-encoder/nli-distilroberta-base"

# Baseline Model path
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_classifier.joblib"

# Retrieval & Verification Thresholds
TOP_K_EVIDENCE = 3
RELEVANCE_THRESHOLD = 0.35  # Cosine similarity threshold for evidence sufficiency
MAX_CLAIMS_PER_ARTICLE = 15

# Canonical Label Definitions
LABEL_SUPPORTED = "SUPPORTED"
LABEL_REFUTED = "REFUTED"
LABEL_UNCERTAIN = "UNCERTAIN"

VALID_LABELS = [LABEL_SUPPORTED, LABEL_REFUTED, LABEL_UNCERTAIN]

# LLM Configuration for Explanation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
