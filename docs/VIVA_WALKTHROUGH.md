# EvidenceLens: Viva and Presentation Walkthrough

This guide follows the implementation in this repository. It distinguishes the working code from broader claims in some project descriptions. The project is an academic prototype for claim-level fact verification; it is not a production news-authenticity oracle.

## 1. One-minute project explanation

> EvidenceLens takes an article, splits it into sentences, keeps candidate factual claims, searches a local evidence corpus for semantically related passages, and checks each claim against those passages with a pretrained Natural Language Inference model. It reports SUPPORTED, REFUTED, or UNCERTAIN, shows the retrieved passages and source links, and gives a short explanation. A separate TF-IDF plus Logistic Regression classifier is included as a conventional Real/Fake baseline. The main pipeline does not train its transformer models in this repository; it loads pretrained models.

## 2. The whole project at a glance

```mermaid
flowchart LR
    A[Article in Streamlit] --> B[Whitespace cleanup]
    B --> C[Regex sentence splitting]
    C --> D[Heuristic claim filter]
    D --> E[MiniLM sentence embeddings]
    E --> F[FAISS IndexFlatIP search]
    F --> G[Top relevant evidence passages]
    G --> H[DistilRoBERTa NLI]
    H --> I[Rule-based multi-passage aggregation]
    I --> J[SUPPORTED / REFUTED / UNCERTAIN]
    J --> K[Grounded explanation]
    K --> L[Streamlit result cards]
```

There are two different tasks in the repository:

| Part | Input and output | Main components | Role |
|---|---|---|---|
| Proposed EvidenceLens pipeline | Article → verdict for each claim | MiniLM + FAISS + NLI verifier | Finds evidence and reasons over claim/evidence pairs |
| Conventional baseline | Whole article → REAL or FAKE | TF-IDF + Logistic Regression | Benchmark comparison only; not used by the proposed pipeline |

## 3. Models and where each is used

### A. `sentence-transformers/all-MiniLM-L6-v2` — evidence search encoder

**Where:** `config.py`, `src/evidence_retrieval/embedding_model.py`, and `src/evidence_retrieval/vector_index.py`.

**What it does:** Converts a claim or a corpus passage into a 384-number vector (an embedding). Texts with related meanings tend to have vectors pointing in similar directions. Corpus passages are embedded when the FAISS index is built; each incoming claim is embedded at search time.

**How its result is used:** Both vectors are L2-normalized. FAISS `IndexFlatIP` ranks passages by inner product, which equals cosine similarity for normalized vectors. The retriever searches up to twice the requested `top_k`, discards scores below `0.20`, removes duplicate passage IDs, and returns at most three passages by default.

**Why this choice:** It is a compact pretrained sentence encoder, produces a fixed-size vector for efficient search, and can match paraphrases better than exact word overlap. The project can run on CPU, with CUDA used when available.

**Alternatives:** TF-IDF/BM25 are simpler and easier to inspect, and can work well when claims and evidence share names or terms, but are weaker for paraphrases. Larger embedding models may retrieve more accurately but cost more memory and time. A hybrid BM25+dense retriever is a reasonable next experiment.

### B. FAISS `IndexFlatIP` — vector search engine, not a learned model

**Where:** `src/evidence_retrieval/vector_index.py`; persisted files are `data/index/evidence_index.faiss` and `data/index/evidence_metadata.jsonl`.

It stores the corpus vectors and searches for the nearest vectors. `Flat` means exact search over all indexed vectors, and `IP` means inner product. It is not trained and does not decide truth. This exact index is appropriate for the small included corpus. For millions of passages, an approximate FAISS index (such as IVF or HNSW) could reduce latency at the cost of search exactness and extra tuning.

### C. `cross-encoder/nli-distilroberta-base` — claim/evidence verifier

**Where:** `config.py`, `src/verification/model_loader.py`, and `src/verification/verifier.py`.

**What it does:** The tokenizer receives an evidence passage and a claim as a paired input. A pretrained DistilRoBERTa sequence classifier returns logits for Natural Language Inference (NLI) relationships: entailment, contradiction, or neutral. The code applies softmax, maps the model's configured label names into project labels, and reports the largest probability as its pair-level label and confidence.

**Mapping:** entailment → SUPPORTED; contradiction → REFUTED; neutral → UNCERTAIN. The model runs in evaluation mode, without gradients, and uses a maximum token length of 256. The implementation checks `id2label` where available and otherwise assumes the common order 0 contradiction, 1 entailment, 2 neutral.

**Why a cross-encoder:** It reads the evidence and claim together, letting attention compare their words and context directly. This is useful for deciding whether one passage supports or contradicts one claim. It is slower than embedding-based retrieval, which is why the project retrieves a small set of candidates first.

**Alternatives:** A bi-encoder can compare precomputed vectors much faster, but usually gives a less precise relation judgment. A larger NLI model might improve accuracy but needs more compute. A fine-tuned domain-specific verifier could fit news verification better, but would require appropriately labeled training data, training code, and a careful evaluation; those are not present for the main verifier here.

### D. TF-IDF + Logistic Regression — conventional baseline

**Where:** `src/evaluation/baseline_classifier.py`, trained by `scripts/train_baseline.py`, serialized to `models/baseline_classifier.joblib`.

TF-IDF converts the whole article into sparse word and two-word-phrase features (up to 5,000 features; English stop words removed; sublinear term-frequency scaling). Logistic Regression learns a linear boundary between `REAL` and `FAKE`. The training script uses a stratified 70/30 split and a fixed random seed of 42.

**Why include it:** It is a familiar, fast, inexpensive reference point that demonstrates the difference between surface-text classification and evidence-grounded verification. It is not called by `FactCheckPipeline` and does not power the article-check tab.

**Alternatives:** Naive Bayes or linear SVM would also be sensible sparse-text baselines. A fine-tuned BERT classifier could model context better, but still would not retrieve evidence or naturally represent lack of evidence, and would need more labeled data and compute. The baseline is a separate binary task, so its metric is not a direct apples-to-apples comparison with the pipeline's three-class FEVER task.

### E. Optional explanation LLM — presentation only, not the judge

**Where:** `src/explanation/explanation_generator.py`.

If `OPENAI_API_KEY` is present, the code attempts an OpenAI Chat Completions request using `OPENAI_MODEL` (default `gpt-4o-mini`) and a low-temperature, evidence-only prompt. If that request fails or no key is configured, it uses a deterministic template that cites the first passage, title, and similarity score. The LLM receives the already-decided label and is instructed not to change it.

**Important accuracy point:** Groq and Gemini keys are read from config and stored by the explanation class, but there are no Groq or Gemini request implementations. The Streamlit help text and architecture docs suggest those integrations more broadly than the code supports. The default fallback is rule-based, not a generative model.

## 4. What happens to one article, step by step

1. **User input:** `app.py` presents the Streamlit interface, sample articles, tabs, and result cards. `st.cache_resource` retains one pipeline instance so the transformer models are not reloaded on each interaction.
2. **Validation and normalization:** `FactCheckPipeline.analyze_article` returns an empty result for blank input. Otherwise, `TextPreprocessor` collapses spaces/tabs, removes zero-width spaces, and normalizes nonbreaking spaces. It preserves the wording rather than stemming or removing stop words.
3. **Sentence splitting:** A regular expression splits on `.`, `!`, or `?` followed by whitespace. A list of common abbreviations is temporarily protected. This is a lightweight tokenizer, not a full linguistic sentence parser.
4. **Claim filtering:** `ClaimExtractor` cleans wrapping quotes, skips duplicates, sentences under four words, questions, short exclamations, and sentences beginning with specified opinion/greeting/call-to-action phrases. It gives surviving sentences heuristic confidence based on a small list of factual signals and a small list of verbs. It keeps at most 15 claims. It does not perform deep entity extraction or semantic claim decomposition.
5. **Claim embedding:** MiniLM encodes each candidate sentence into a normalized 384-dimensional vector.
6. **Evidence retrieval:** FAISS searches the indexed corpus for the nearest vectors. Results below the 0.20 relevance threshold are filtered. Three passages are returned by default.
7. **Pairwise NLI:** The verifier evaluates every returned passage against the claim. If no passage survived retrieval, the verdict is immediately UNCERTAIN with confidence 0 and reason `missing_evidence`.
8. **Passage aggregation:** With any passage of relevance at least 0.60, the code looks at labels from all such high-relevance passages. If both support and refutation appear, it assigns UNCERTAIN with a conflict flag and confidence 0.50. Otherwise, it uses the top-ranked passage's label and confidence. If none reaches 0.60, it relevance-weights the three-class probabilities across the returned passages and takes the largest average.
9. **Explanation:** The rule-based explainer or optional OpenAI call produces a short explanation from the verdict and evidence.
10. **Result assembly:** Each claim result includes its ID, text, verdict, confidence, retrieval status, passages, explanation, verifier metadata, original sentence position, and extraction heuristic score. The article summary counts labels and records elapsed time. There is no single article-level REAL/FAKE verdict.

## 5. Repository map: what each portion is for

| Path | Purpose |
|---|---|
| `app.py` | Streamlit interface: article input, sample stories, claim result cards, metrics, system explanation, and limitations. |
| `config.py` | Central paths, model names, label constants, thresholds, and environment-variable loading. |
| `src/preprocessing/` | Whitespace normalization, approximate sentence segmentation, and light quote cleanup. |
| `src/claim_extraction/` | Heuristic sentence filtering, IDs, positions, and extraction-confidence heuristic. |
| `src/evidence_retrieval/` | JSONL evidence loading; MiniLM encoding; FAISS build/save/load/search; relevance filtering. |
| `src/verification/` | Transformer loading/cache, NLI inference, canonical label mapping, and multi-passage aggregation. |
| `src/explanation/` | Evidence-grounded deterministic explanations and optional OpenAI explanation request. |
| `src/pipeline/` | Orchestrates the stages, catches per-claim exceptions, and assembles output/summary. |
| `src/evaluation/` | Baseline classifier, FEVER metrics, and comparison report generation. |
| `src/utils/` | Logging and a text-hashing helper. The hash helper is not used in the main pipeline. |
| `scripts/prepare_dataset.py` | Maps FEVER labels to project labels and writes processed JSONL. |
| `scripts/build_evidence_index.py` | Embeds evidence passages and saves the FAISS index plus metadata. |
| `scripts/train_baseline.py` | Trains/evaluates/saves the TF-IDF + Logistic Regression baseline. |
| `scripts/run_evaluation.py` | Runs the main pipeline over processed FEVER records and writes reports. |
| `data/sample/` | Small sample FEVER claims, article-classification examples, and evidence passages. |
| `data/processed/` | Processed FEVER records used by the included evaluation. |
| `data/index/` | Saved vector index and passage metadata, loaded by the application. |
| `models/` | Saved baseline classifier artifact; the two transformer models are fetched by Hugging Face model IDs and cached by their libraries, not stored here as project checkpoints. |
| `reports/` | Checked-in metric snapshots and their comparative write-up. |
| `tests/` | Unit/integration checks for preprocessing, extraction, retrieval, verification, explanation, and orchestration. |
| `docs/` | Architecture, methodology, dataset, evaluation, limitations, and viva material. |

`requirements.txt` lists the Python libraries: PyTorch/Transformers/Sentence-Transformers for inference, FAISS for vector search, scikit-learn for the baseline and metrics, Streamlit for the UI, plus data/configuration utilities.

## 6. Data and preparation

- **FEVER** is the claim-verification benchmark format used here. Preparation maps SUPPORTS → SUPPORTED, REFUTES → REFUTED, and NOT ENOUGH INFO → UNCERTAIN. The included `fever_sample.jsonl` and processed file contain 20 records, not the full 185,445-record FEVER dataset described in some documentation.
- **Evidence corpus:** JSONL records provide a passage ID, title, text, source ID, and source URL. `DocumentLoader` skips malformed JSON lines and records without passage text. The included corpus has 20 evidence passages.
- **Index build:** `build_evidence_index.py` embeds each `title: text` string, adds normalized vectors to a 384-dimensional exact FAISS index, then stores metadata separately so search results can be displayed with source details.
- **Important distinction:** The main transformer models are pretrained inference components. The repository prepares a small evaluation dataset and trains only the conventional baseline; it does not fine-tune MiniLM or DistilRoBERTa.

## 7. Evaluation: what the stored results say and do not say

The checked-in `reports/pipeline_metrics.json` reports 20 FEVER claims: accuracy 0.75 and macro-F1 0.7342. It reports Recall@k 1.0 and MRR 1.0 over the 15 records that have evidence references. The retrieval evaluator matches reference titles/passages to the retrieved set; the metric is not proof that retrieval works on arbitrary real-world news.

The baseline report evaluates 8 held-out articles: accuracy 0.625 and macro-F1 0.5636. Because the baseline and pipeline use different datasets, sample sizes, input units, and label sets, avoid saying the proposed model is proven “better” based only on these numbers. Say that these are preliminary results on the included small samples, while the pipeline's differentiator is evidence retrieval and claim-level citations.

The FEVER sample and evidence corpus are tiny and curated around overlapping topics. Results may be optimistic and should not be generalized. Softmax confidence is model confidence over its output classes, not a calibrated probability that a statement is true in the world.

## 8. Design choices and alternatives: concise viva answers

**Why separate retrieval and verification?** Retrieval narrows a large corpus to a few candidates efficiently; the cross-encoder then spends more computation on detailed comparisons. A single generative model answering from memory would not give the same controlled indexed evidence trail.

**Why use three labels?** SUPPORT/REFUTE alone forces a decision when evidence is missing. UNCERTAIN preserves the difference between contradiction and lack of sufficient evidence.

**Why not simply use a large LLM to detect fake news?** It may answer from parametric memory, hallucinate citations, and provide inconsistent judgments. Here a verifier assigns the label against retrieved passages; an optional LLM only phrases an explanation. Human review and better evidence coverage are still needed.

**Could claim extraction be improved?** Yes. Dependency parsing, semantic role labeling, or a trained claim detector could detect claims more precisely and split compound sentences, but they introduce model/runtime costs and need evaluation data. The current regex/heuristic approach is lightweight and easy to demonstrate, but can miss claims and retain opinions.

**Could retrieval be improved?** Yes: BM25+dense hybrid search, a larger encoder, passage chunking, reranking, source authority, and timestamps are plausible. They require larger corpora, careful threshold calibration, and evaluation for recall, precision, latency, and freshness.

**Could the verifier be improved?** Yes: calibrate thresholds on a validation set, evaluate a stronger NLI model, train on domain-specific annotated pairs, or combine evidence across sentences more robustly. Do not claim the current aggregation performs logical proof: it is a small heuristic around model outputs and similarity values.

**Why a local/static evidence corpus?** It makes runs reproducible and limits uncontrolled web noise. The tradeoff is coverage and freshness: new, local, or obscure facts may be absent and should remain UNCERTAIN.

## 9. Limitations and responsible interpretation

1. The corpus is static and small; retrieval failure means the system cannot verify the claim from its available sources.
2. Sentence segmentation and claim filtering are heuristic. Complex claims, sarcasm, quotations, and mixed factual/opinion sentences can be mishandled.
3. Similarity is relevance, not truth. A high score only says the passage is semantically related.
4. NLI models can make errors, and their softmax scores are not calibrated truth probabilities.
5. The multi-passage logic is heuristic and may be sensitive to retrieval order, relevance cutoffs, and false NLI predictions.
6. Sources in the sample corpus are Wikipedia-based; no robust source credibility or recency ranking is implemented.
7. LLM explanations may still add unsupported phrasing despite prompt constraints; the deterministic mode is safer but templated.
8. The tool checks claims, not the intent of a publisher or the overall credibility of an article. Treat output as decision support, not a final fact-check.

## 10. Suggested live demonstration

1. Start the Streamlit app and select the mixed sample article.
2. Show that one article can yield different outcomes for different sentences; there is deliberately no one-word verdict for the entire article.
3. Expand a claim and point to its passage text, title/source URL, and semantic relevance score.
4. Explain the model chain: MiniLM finds candidates; FAISS ranks stored vectors; DistilRoBERTa NLI judges each evidence/claim pair; the aggregation rules form the result.
5. Select an unsupported claim and explain why no retrieved passage leads to UNCERTAIN rather than REFUTED.
6. Open the benchmark tab, clearly stating that the stored pipeline score uses 20 claims and baseline score uses 8 test articles.

## 11. A short presentation script

> Our project is EvidenceLens, a prototype for evidence-grounded fact verification. Rather than classify an entire article directly as real or fake, it extracts candidate factual sentences and checks each separately. We use all-MiniLM-L6-v2 to encode claims and evidence passages for semantic search, FAISS IndexFlatIP to retrieve the closest passages, and a DistilRoBERTa NLI cross-encoder to classify each evidence/claim pair as entailment, contradiction, or neutral. These become SUPPORTED, REFUTED, and UNCERTAIN. If the corpus has no relevant evidence, the system returns UNCERTAIN instead of assuming the claim is false. Results include the passages and source information for review. A separate TF-IDF plus Logistic Regression model is our conventional binary baseline, not part of the proposed pipeline. Our current evaluation is preliminary: it uses 20 FEVER claims and an 8-article baseline test split, so larger and more diverse evaluation is necessary.

## 12. Source files to open during questions

- Start with `src/pipeline/fact_check_pipeline.py` for end-to-end orchestration.
- Show `src/claim_extraction/claim_extractor.py` for the heuristic extraction rules.
- Show `src/evidence_retrieval/embedding_model.py` and `vector_index.py` for MiniLM + FAISS.
- Show `src/verification/verifier.py` for NLI, label mapping, and aggregation.
- Show `src/explanation/explanation_generator.py` for the deterministic/optional OpenAI explanation.
- Show `src/evaluation/baseline_classifier.py` for TF-IDF + Logistic Regression.
- Show `config.py` for the exact model names and thresholds.
- Show `reports/pipeline_metrics.json` and `reports/baseline_metrics.json` for the stored evaluation snapshots.
