# 📚 EvidenceLens — Complete Project Study Guide
**For Project Evaluation | B.Tech CSE Sem 5 | CSBB 311 | NIT Delhi**
**Supervisor: Dr. Gautam Kumar**

---

## 👥 Team Members & Roles

| Name | Roll No. | Primary Area |
|:---|:---|:---|
| **Krrish Kumar** | 241210070 | Pipeline Orchestration & Testing |
| **Mayank** | 241210075 | Evidence Retrieval (FAISS + Embeddings) |
| **Mishti Gupta** | 241210079 | Claim Extraction & Preprocessing |
| **Prakshi Agarwal** | 241210091 | Verification Module & Label Mapping |
| **Rachit Singh Rana** | 241210097 | Evaluation, Baseline & Web App (Streamlit) |

> **Note:** The GitHub repo is under Mayank's account (`Mayank-coder30/Fake-News-Detection`). Each member can explain any module during the viva — see the "What You Can Claim" section below.

---

## 🧠 What Is This Project?

**EvidenceLens** is an **Evidence-Grounded Fake News Detection & Fact Verification System**.

### The Core Problem
Traditional fake news detectors are simple binary text classifiers:
- They look at writing *style* (clickbait words, punctuation, emotional tone)
- They output just "REAL" or "FAKE"
- They can't tell you *why* something is fake
- A professionally written fake article easily fools them

### Our Solution
Instead of classifying style, we verify **facts**:
1. Break the article into individual factual **claims**
2. **Retrieve evidence** from a knowledge corpus using semantic search
3. **Verify** each claim against that evidence using AI
4. Output **SUPPORTED / REFUTED / UNCERTAIN** with full citations

---

## 🏗️ System Architecture — The 4-Stage Pipeline

```
News Article Input
      │
      ▼
┌─────────────────────────────────┐
│  STAGE 1: Preprocessing         │
│  - Clean text, fix whitespace   │
│  - Segment into sentences       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  STAGE 2: Claim Extraction      │
│  - Filter opinions & questions  │
│  - Keep declarative facts only  │
│  - Score by entity/date/number  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  STAGE 3: Evidence Retrieval    │
│  - Encode claim → 384-dim vector│
│  - FAISS search Wikipedia corpus│
│  - Return Top-3 passages        │
│  - Filter by threshold (≥ 0.20) │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  STAGE 4: NLI Verification      │
│  - DistilRoBERTa Cross-Encoder  │
│  - [CLS] Evidence [SEP] Claim   │
│  - Output: SUPPORTED / REFUTED  │
│           / UNCERTAIN           │
└──────────────┬──────────────────┘
               │
               ▼
        Explanation + UI
```

---

## 🔧 Technologies & Libraries Used

| Technology | Version | Purpose |
|:---|:---|:---|
| **Python** | 3.10 | Core language |
| **PyTorch** | ≥ 2.0.0 | Deep learning backend |
| **Transformers** (HuggingFace) | ≥ 4.35.0 | NLI cross-encoder model |
| **Sentence-Transformers** | ≥ 2.2.2 | Text embedding model |
| **FAISS-CPU** | ≥ 1.7.4 | Vector similarity search |
| **Scikit-Learn** | ≥ 1.3.0 | Baseline TF-IDF + LogReg |
| **Streamlit** | ≥ 1.28.0 | Interactive web dashboard |
| **NLTK** | ≥ 3.8.1 | Sentence tokenization |
| **Pandas / NumPy** | ≥ 2.0 / 1.24 | Data handling |
| **Pydantic** | ≥ 2.0.0 | Data validation/schemas |
| **Pytest** | ≥ 7.4.0 | Testing framework |

---

## 🤖 AI Models Used

### Model 1: `sentence-transformers/all-MiniLM-L6-v2`
- **Purpose**: Converts claims and evidence passages into 384-dimensional vectors
- **Why**: Can find semantically similar text even with different wording (e.g. "infrared astronomy" vs "observes in the infrared spectrum")
- **How**: Trained on over 1 billion sentence pairs for semantic similarity

### Model 2: `cross-encoder/nli-distilroberta-base`
- **Purpose**: Verifies whether evidence *supports*, *refutes*, or is *neutral* to a claim
- **Why**: Cross-encoders do full bidirectional attention over both texts simultaneously — much more powerful than simple classifiers
- **Input format**: `[CLS] evidence_text [SEP] claim_text [SEP]`
- **Output**: 3 logits → softmax → SUPPORTED / REFUTED / UNCERTAIN

### Baseline: TF-IDF + Logistic Regression
- **Purpose**: A conventional "dumb" classifier for comparison
- **Input**: Full article text
- **Output**: Binary REAL / FAKE label
- **Limitation**: Zero evidence, zero explanation

---

## 📊 Dataset: FEVER

**FEVER = Fact Extraction and VERification**
- Published at NAACL-HLT 2018 (Thorne et al.)
- 185,445 claims generated from Wikipedia
- Each claim labeled: `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO`
- We used a curated sample of 20 claims for benchmark evaluation

---

## 📈 Results at a Glance

### Our Pipeline (Evidence-Grounded)
| Metric | Value |
|:---|:---|
| Accuracy | **75.00%** |
| Macro F1 | **0.7342** |
| Evidence Recall@3 | **100%** (1.0000) |
| Mean Reciprocal Rank | **1.0000** |

### Baseline (TF-IDF + LogReg)
| Metric | Value |
|:---|:---|
| Accuracy | **62.50%** |
| Macro F1 | **0.5636** |
| Evidence Citations | **None (0%)** |
| Epistemic Uncertainty | **None** |

### Per-Class Breakdown (Our System)
| Class | Precision | Recall | F1 |
|:---|:---|:---|:---|
| SUPPORTED | 1.0000 | 0.6667 | **0.8000** |
| REFUTED | 0.7500 | 1.0000 | **0.8571** |
| UNCERTAIN | 0.5000 | 0.6000 | **0.5455** |

> **Key talking point**: We improved accuracy by **+12.5%** over baseline AND added full explainability.

---

## 📁 Project File Structure Explained

```
Fake-News-Detection/
│
├── app.py              → Streamlit web UI (5 tabs: verify, benchmark, how it works, info, limitations)
├── config.py           → All settings: paths, model names, thresholds, API keys
├── requirements.txt    → 14 libraries listed
│
├── src/                → All source code, organized by module
│   ├── preprocessing/  → Cleans text, splits into sentences
│   ├── claim_extraction/ → Filters factual claims from text
│   ├── evidence_retrieval/ → FAISS + embeddings for semantic search
│   ├── verification/   → NLI transformer verifier
│   ├── explanation/    → Generates human-readable explanation
│   ├── evaluation/     → Baseline model + metrics computation
│   ├── pipeline/       → Connects all 4 stages end-to-end
│   └── utils/          → Logging + caching utilities
│
├── scripts/            → One-time setup scripts
│   ├── prepare_dataset.py   → Standardize FEVER data
│   ├── build_evidence_index.py → Build FAISS vector index
│   ├── train_baseline.py   → Train TF-IDF+LogReg baseline
│   └── run_evaluation.py   → Compute all metrics
│
├── tests/              → 26 unit & integration tests (all pass ✅)
├── data/               → Sample corpus, processed data, FAISS index files
├── models/             → Saved baseline model (.joblib)
├── reports/            → JSON metrics + comparison report
└── docs/               → Architecture, Methodology, Evaluation, Limitations, VIVA prep
```

---

## 🧩 Module-by-Module Breakdown

### `src/preprocessing/text_preprocessor.py`
- Normalizes whitespace and unicode
- Splits text into sentences (handles `Dr.`, `Prof.`, `U.S.` without false splits)
- Cleans quotes and formatting

### `src/claim_extraction/claim_extractor.py`
- **Filters OUT**: questions (`?`), opinions (`"I believe"`, `"In my opinion"`), greetings, short sentences
- **Keeps**: declarative sentences with entities, dates, numbers, factual verbs (`announced`, `discovered`)
- Assigns each claim a unique ID like `claim_001`, a position, and a confidence score

### `src/evidence_retrieval/`
- `document_loader.py` — Reads evidence corpus from `.jsonl` files
- `embedding_model.py` — Encodes text using `all-MiniLM-L6-v2`, normalizes vectors to unit length
- `vector_index.py` — FAISS `IndexFlatIP` for cosine similarity search
- `evidence_retriever.py` — For each claim, finds top-3 passages, filters those below 0.20 threshold

### `src/verification/`
- `verifier.py` — Sends `(evidence, claim)` pairs to `nli-distilroberta-base`
- `label_mapping.py` — Maps FEVER labels (`SUPPORTS` → `SUPPORTED`, `NOT ENOUGH INFO` → `UNCERTAIN`)
- **Multi-Evidence Aggregation logic**:
  - If top passage has similarity ≥ 0.60 → it's the primary anchor
  - If both SUPPORT and REFUTE signals → mark as UNCERTAIN with conflict warning
  - If no passage above threshold → UNCERTAIN (never falsely REFUTED)

### `src/explanation/explanation_generator.py`
- **Deterministic mode (default)**: Quotes the actual evidence passage, source title, and relevance score
- **Optional LLM mode**: Calls OpenAI/Groq/Gemini API with tight anti-hallucination prompts
- LLM **cannot change the verdict** — it only rephrases the explanation

### `src/pipeline/fact_check_pipeline.py`
- 12-step orchestration: input validation → preprocessing → claim extraction → retrieval → verification → explanation → aggregation → JSON output
- Returns a `PipelineResult` with `claims[]` and a `summary` (counts + timing)

### `src/evaluation/`
- `baseline_classifier.py` — TF-IDF (1-2 n-grams, 5000 features) + Logistic Regression
- `evaluate_verifier.py` — Computes Accuracy, Precision, Recall, F1, Confusion Matrix, Recall@k, MRR
- `compare_models.py` — Generates the comparison Markdown report

---

## 🔑 Key Concepts to Know Cold

### What is NLI (Natural Language Inference)?
A 3-way classification task: Given a **premise** (evidence) and a **hypothesis** (claim), does the premise:
- **Entail** the hypothesis? → SUPPORTED
- **Contradict** the hypothesis? → REFUTED
- **Neither** (neutral)? → UNCERTAIN

### What is FAISS?
Facebook AI Similarity Search — a library for fast nearest-neighbor search in high-dimensional vector spaces. We use `IndexFlatIP` (Inner Product on unit-normalized vectors = cosine similarity).

### Why normalize vectors before FAISS?
When all vectors have unit L2 norm (||e|| = 1), the dot product equals cosine similarity:
$$\langle \hat{e}_c, \hat{e}_p \rangle = \cos(e_c, e_p)$$

### Why Sentence-Transformers over TF-IDF for retrieval?
TF-IDF does exact keyword matching. Sentence-Transformers create **dense semantic representations** — meaning similar *ideas* are close in vector space even with different words.

### What is the Relevance Threshold (0.20)?
Before passing evidence to the NLI verifier, we check if the semantic similarity score ≥ 0.20. Below this, the claim is immediately marked **UNCERTAIN** because the retrieved "evidence" is likely irrelevant.

### What is Macro-F1?
The unweighted average F1 across all 3 classes:
$$\text{Macro-F1} = \frac{F1_{\text{SUPPORTED}} + F1_{\text{REFUTED}} + F1_{\text{UNCERTAIN}}}{3}$$
Better than accuracy because it equally weights minority classes.

### What is Recall@k and MRR?
- **Recall@k**: Did at least 1 ground-truth evidence passage appear in the top-k results? We got 100%.
- **MRR**: Average of the reciprocal rank of the first correct result. Score of 1.0 means the correct passage was always rank #1.

---

## 🚨 Limitations (Know These!)

1. **Static Corpus**: Only knows what's in the indexed knowledge base. Events after indexing → UNCERTAIN (correct behavior)
2. **Temporal Drift**: Old passages may have outdated facts (e.g., who is currently president)
3. **Claim Extraction Misses**: Subtle or conditional claims may be filtered or missed
4. **Confidence ≠ Truth**: 90% model confidence means strong token alignment, NOT guaranteed real-world truth
5. **LLM Hallucination Risk**: If LLM API is used for explanation, it could embellish — mitigated by low temperature (0.2), strict prompts, and deterministic fallback

---

## 🧪 Testing (26/26 Passed)

| Test File | What It Tests |
|:---|:---|
| `test_preprocessing.py` | Whitespace normalization, sentence segmentation, empty inputs |
| `test_claim_extraction.py` | Claim filtering, opinion removal, deduplication, edge cases |
| `test_evidence_retrieval.py` | Document loading, embedding encoding, FAISS build & search |
| `test_verification.py` | Label mapping, missing evidence → UNCERTAIN, pairwise NLI |
| `test_explanations.py` | Explanation generation for all 4 verdict types |
| `test_pipeline.py` | End-to-end: empty article, supported, refuted, uncertain, summary counts |

---

## 🎤 Expected Viva Questions & Answers

**Q: What problem does this project solve?**
A: Traditional fake news detectors use style/word-frequency classifiers that can't verify actual facts. We built an evidence-grounded system that decomposes articles into claims, retrieves Wikipedia evidence, and uses NLI to classify each claim as SUPPORTED, REFUTED, or UNCERTAIN with full citations.

**Q: Why is your approach better than TF-IDF + Logistic Regression?**
A: Our system: (1) verifies actual factual content not style, (2) provides evidence citations for every prediction, (3) handles epistemic uncertainty instead of forcing a binary guess, (4) achieves 75% accuracy vs 62.5% for baseline.

**Q: Why did you use FAISS?**
A: FAISS enables millisecond-scale cosine similarity search over dense 384-dimensional vectors. For CPU deployment without a database server, it's the industry-standard choice for semantic retrieval at scale.

**Q: What is a cross-encoder? Why not just use a bi-encoder?**
A: A cross-encoder processes the full concatenated `[CLS] Evidence [SEP] Claim [SEP]` through bidirectional attention, letting every word in the claim attend to every word in the evidence. Bi-encoders encode texts independently and lose this interaction. Cross-encoders are more accurate but slower — acceptable for our per-claim verification use case.

**Q: How do you handle the case where evidence is absent?**
A: Critical principle: *"Absence of evidence is NOT evidence of absence."* If the relevance score of retrieved passages is below 0.20, we classify the claim as UNCERTAIN, never REFUTED. This is an ethical safety guarantee built into the pipeline.

**Q: What did you actually implement vs what was pre-trained?**
A: Pre-trained (reused): `all-MiniLM-L6-v2` and `nli-distilroberta-base`.
Implemented by team: the entire modular pipeline, claim extractor, FAISS index management, multi-evidence aggregation logic, deterministic explanation engine, baseline classifier, evaluation suite, Streamlit UI, and all 26 tests.

**Q: Why did UNCERTAIN class perform worst (F1 = 0.5455)?**
A: UNCERTAIN is the hardest class — it captures both "no evidence found" AND "evidence is genuinely inconclusive." Some claims the model labeled UNCERTAIN were actually REFUTED by the evidence corpus (e.g., Einstein's cat claim). This reflects the fundamental difficulty of distinguishing "no knowledge" from "no support."

**Q: What are the ethical considerations?**
A: (1) System is a research prototype, not a production fact-checker. (2) UNCERTAIN ≠ false — users must not misinterpret lack of evidence as proof of falsehood. (3) Knowledge corpus has a cutoff date. (4) Sarcasm and nuanced language may be misclassified.

---

## 💡 Demo Tips for Tomorrow

When presenting to Dr. Gautam Kumar:

1. **Run the app**: `streamlit run app.py` → open `http://localhost:8501`
2. **Best demo**: Select **"Sample 4: Mixed News Story (Multi-Verdict)"**
   - JWST claim → SUPPORTED (with Wikipedia citation)
   - Apollo 11 to Mars → REFUTED (with contradicting evidence)
   - Einstein's cat → UNCERTAIN (no evidence found)
3. **Compare tabs**: Show the "Benchmark & Comparison" tab to highlight 75% vs 62.5%
4. **Point out**: The evidence box with relevance scores — this is the key differentiator
5. **Mention the 26 tests**: Shows engineering rigor

---

## 📌 Team Division of Work (Suggested Explanation)

> You can frame this naturally during the viva:

- **Mishti** handled the **NLP preprocessing and claim extraction** — figuring out which sentences are actually factual claims vs opinions.
- **Mayank** worked on the **semantic retrieval system** — setting up FAISS, integrating Sentence-Transformers, building and persisting the evidence index.
- **Prakshi** implemented the **NLI verification module** — integrating DistilRoBERTa, implementing the multi-evidence aggregation logic, and the label mapping system.
- **Krrish** built the **end-to-end pipeline orchestration and the test suite** — connecting all modules together and writing the 26 automated tests.
- **Rachit** implemented the **evaluation framework and the Streamlit web app** — training the baseline, computing metrics, building the interactive dashboard, and generating reports.

All members collaborated on the architecture design, documentation, and final integration.
