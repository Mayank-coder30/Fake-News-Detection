# Comparative Evaluation Report: Conventional Baselines vs. Evidence-Grounded Pipeline
**Project**: Evidence-Grounded Fake News Detection and Fact Verification (EvidenceLens)  
**Academic Institution**: National Institute of Technology Delhi (CSBB 311)  
**Supervisor**: Dr. Gautam Kumar  

---

## 1. Executive Summary

This benchmark rigorously contrasts two paradigms in automated misinformation detection:
1. **Conventional Surface-Level Baselines**: Uses stylistic/lexical patterns (TF-IDF n-grams + Logistic Regression) without external knowledge retrieval.
2. **Evidence-Grounded Verification Pipeline (Proposed)**: Decomposes articles into checkable claims, retrieves candidate evidence passages from an indexed knowledge corpus via dense semantic embeddings (Sentence-Transformers + FAISS), and conducts pairwise Transformer NLI inference to classify claims into **SUPPORTED**, **REFUTED**, or **UNCERTAIN**, with transparent natural-language explanations.

---

## 2. Quantitative Performance Comparison

| Metric / Dimension | Conventional Baseline (FEVER Claims) | Conventional Baseline (News Articles) | EvidenceLens Proposed Pipeline |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | TF-IDF (1-2 n-grams) + Logistic Regression | TF-IDF (1-2 n-grams) + Logistic Regression | Sentence-Transformers + FAISS + Transformer NLI |
| **Input Unit** | Factual Claim Sentences | Whole Article Text | Granular Factual Claims |
| **Knowledge Grounding** | None (Closed-world surface lexical patterns) | None (Closed-world surface lexical patterns) | Open-world Retrieval from 586-passage Verified Corpus |
| **Target Taxonomy** | 3-Class (`SUPPORTED`, `REFUTED`, `UNCERTAIN`) | 2-Class (`REAL`, `FAKE`) | 3-Class (`SUPPORTED`, `REFUTED`, `UNCERTAIN`) |
| **Evaluation Dataset** | FEVER Benchmark Test Split (N=240) | News Articles Test Split (N=31) | FEVER Benchmark Verification Split (N=800) |
| **Accuracy** | **34.58%** | **96.77%** | **47.88%** |
| **Macro F1-Score** | **0.3403** | **0.9676** | **0.4340** |
| **Evidence Recall@k** | N/A (No retrieval capability) | N/A (No retrieval capability) | **0.1592** |
| **Mean Reciprocal Rank (MRR)** | N/A (No retrieval capability) | N/A (No retrieval capability) | **0.1526** |
| **Auditability & Explanations** | Black-box decision based on word frequencies | Black-box decision based on word frequencies | Transparent evidence passages + Grounded explanation |
| **Handles Missing Evidence?** | No (Forces prediction without grounding) | No (Forces binary Real or Fake prediction) | **Yes** (Safely predicts `UNCERTAIN`) |

---

## 3. Scientific Discussion & Findings

### Why the Surface Baseline Fails on FEVER (34.58% Accuracy)
When trained directly on factual claim sentences from FEVER, the conventional TF-IDF baseline achieves only **34.58% accuracy** (equivalent to random guessing on a balanced 3-class problem). 
This demonstrates an essential scientific principle: **the veracity of a factual proposition cannot be determined from word frequencies alone**. 
For example:
- *"The James Webb Space Telescope conducts infrared astronomy."* (True)
- *"The James Webb Space Telescope was launched in 1995."* (False)

Both statements share virtually identical vocabulary and grammar. Without consulting external reference evidence, a statistical n-gram model has zero basis to distinguish between them.

### EvidenceLens Result (47.88% Accuracy vs. 34.58% Baseline)
By pairing dense semantic retrieval (FAISS) over an expanded corpus with cross-encoder neural entailment (DistilRoBERTa NLI):
1. The pipeline retrieves authentic Wikipedia evidence documents corresponding to entities in the claims.
2. The cross-encoder directly measures semantic entailment and contradiction against verified ground truth.
3. When evidence is insufficient, the system safely falls back to **UNCERTAIN**, avoiding fabricated predictions.

On the evaluation dataset (N=800), EvidenceLens achieves **47.88% accuracy** and **0.4340 macro F1**, significantly outperforming the conventional surface baseline (**34.58%**). Notably, when EvidenceLens predicts **SUPPORTED**, it attains **92.31% precision**, reflecting high reliability and epistemic safety.

---

## 4. Detailed Breakdown of Proposed Pipeline

### Per-Class Metrics (FEVER Benchmark):

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **SUPPORTED** | 0.9231 | 0.1348 | 0.2353 | 267 |
| **REFUTED** | 0.5429 | 0.4981 | 0.5195 | 267 |
| **UNCERTAIN** | 0.4147 | 0.8045 | 0.5473 | 266 |

### Confusion Matrix:
Labels: `['SUPPORTED', 'REFUTED', 'UNCERTAIN']`
```
[36, 61, 170]
[2, 133, 132]
[1, 51, 214]
```
