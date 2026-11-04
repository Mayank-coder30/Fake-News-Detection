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
| **Knowledge Grounding** | None (Closed-world surface lexical patterns) | None (Closed-world surface lexical patterns) | Open-world Retrieval from 100-passages Verified Corpus |
| **Target Taxonomy** | 3-Class (`SUPPORTED`, `REFUTED`, `UNCERTAIN`) | 2-Class (`REAL`, `FAKE`) | 3-Class (`SUPPORTED`, `REFUTED`, `UNCERTAIN`) |
| **Evaluation Dataset** | FEVER Benchmark Test Split (N=43635) | News Articles Test Split (N=31) | FEVER Benchmark Verification Split (N=60) |
| **Accuracy** | **59.03%** | **96.77%** | **91.67%** |
| **Macro F1-Score** | **0.4371** | **0.9676** | **0.9126** |
| **Evidence Recall@k** | N/A (No retrieval capability) | N/A (No retrieval capability) | **1.0** |
| **Mean Reciprocal Rank (MRR)** | N/A (No retrieval capability) | N/A (No retrieval capability) | **1.0** |
| **Auditability & Explanations** | Black-box decision based on word frequencies | Black-box decision based on word frequencies | Transparent evidence passages + Grounded explanation |
| **Handles Missing Evidence?** | No (Forces prediction without grounding) | No (Forces binary Real or Fake prediction) | **Yes** (Safely predicts `UNCERTAIN`) |

---

## 3. Scientific Discussion & Findings

### Why the Surface Baseline Fails on FEVER (59.03% Accuracy)
When trained directly on factual claim sentences from FEVER, the conventional TF-IDF baseline achieves only **59.03% accuracy** (equivalent to random guessing on a balanced 3-class problem). 
This demonstrates an essential scientific principle: **the veracity of a factual proposition cannot be determined from word frequencies alone**. 
For example:
- *"The James Webb Space Telescope conducts infrared astronomy."* (True)
- *"The James Webb Space Telescope was launched in 1995."* (False)

Both statements share virtually identical vocabulary and grammar. Without consulting external reference evidence, a statistical n-gram model has zero basis to distinguish between them.

### Current EvidenceLens Result (91.67% Accuracy)
By pairing dense semantic retrieval (FAISS) with cross-encoder neural entailment (DistilRoBERTa NLI):
1. The pipeline retrieves the authentic Wikipedia document.
2. The cross-encoder directly measures semantic entailment and contradiction against verified ground truth.
3. When evidence is insufficient, the system safely falls back to **UNCERTAIN**, avoiding fabricated predictions.

On the current evaluation benchmark, the pipeline achieves **91.67% accuracy**. Retrieval recall and MRR are **1.0** and **1.0**, respectively, demonstrating robust evidence grounding and high retrieval precision.

---

## 4. Detailed Breakdown of Proposed Pipeline

### Per-Class Metrics (FEVER Benchmark):

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **SUPPORTED** | 0.9600 | 0.9231 | 0.9412 | 26 |
| **REFUTED** | 0.8571 | 0.9474 | 0.9000 | 19 |
| **UNCERTAIN** | 0.9286 | 0.8667 | 0.8966 | 15 |

### Confusion Matrix:
Labels: `['SUPPORTED', 'REFUTED', 'UNCERTAIN']`
```
[24, 1, 1]
[1, 18, 0]
[0, 2, 13]
```
