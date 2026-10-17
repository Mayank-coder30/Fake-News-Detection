# Comparative Evaluation Report: Conventional Baseline vs. Evidence-Grounded Pipeline
**Project**: Evidence-Grounded Fake News Detection and Fact Verification (EvidenceLens)  
**Academic Institution**: National Institute of Technology Delhi (CSBB 311)

---

## 1. Executive Summary

This evaluation contrasts two fundamentally distinct paradigms in automated misinformation detection:
1. **Conventional Text Classification Baseline**: Uses stylistic/lexical patterns (TF-IDF n-grams + Logistic Regression) to assign coarse binary Real/Fake labels without external verification.
2. **Evidence-Grounded Verification Pipeline (Proposed)**: Decomposes articles into checkable claims, retrieves candidate evidence passages from a indexed knowledge corpus via dense semantic embeddings (Sentence-Transformers + FAISS), and conducts pairwise Transformer NLI inference to classify claims into **SUPPORTED**, **REFUTED**, or **UNCERTAIN**, with transparent natural-language explanations.

---

## 2. Quantitative Performance Comparison

| Metric / Attribute | Conventional Text Classifier (Baseline) | Evidence-Grounded Pipeline (Proposed) |
| :--- | :--- | :--- |
| **Model Architecture** | TF-IDF (1-2 ngrams) + Logistic Regression | Sentence-Transformers + FAISS + Transformer NLI |
| **Input Unit** | Entire news article / headline | Granular, individual factual claims |
| **Knowledge Grounding** | None (Closed-world surface lexical patterns) | Open-world Retrieval from Verified Knowledge Corpus |
| **Target Taxonomy** | 2-Class (`REAL`, `FAKE`) | 3-Class (`SUPPORTED`, `REFUTED`, `UNCERTAIN`) |
| **Evaluation Dataset** | News Articles Held-out Split (N=8) | FEVER Benchmark Verification Split (N=20) |
| **Accuracy** | **0.625** | **0.75** |
| **Macro F1-Score** | **0.5636** | **0.7342** |
| **Evidence Recall@k** | N/A (No retrieval capability) | **1.0** |
| **Mean Reciprocal Rank (MRR)** | N/A (No retrieval capability) | **1.0** |
| **Auditability / Interpretability** | Black-box decision based on word frequencies | Transparent evidence passages + Grounded explanation |
| **Handles Missing Evidence?** | No (Forces binary Real or Fake prediction) | **Yes** (Safely predicts `UNCERTAIN`) |

---

## 3. Methodological Distinctions & Discussion

### A. Why Conventional Classification Fails in Practice
Conventional classifiers detect superficial stylistic proxies (sensationalist words, punctuation, emotional rhetoric). When applied to plausible, professionally written misinformation or novel breaking news, surface text classifiers suffer severe distribution shift and fail to determine whether the factual claims are true.

### B. The Advantage of the Evidence-Grounded Approach
By grounding each claim in retrieved evidence passages:
1. **Verifiable Audit Trail**: Journalists, editors, and students can review the exact source passage and relevance score.
2. **Epistemic Modality (UNCERTAIN)**: Distinguishes between claims that are actively refuted by contradictory evidence versus claims where evidence is currently unavailable or inconclusive.
3. **No Fabricated Labels**: A low retrieval score triggers `UNCERTAIN` rather than false certainty.

---

## 4. Detailed Breakdown of Proposed Pipeline

### Per-Class Metrics (FEVER Benchmark):

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **SUPPORTED** | 1.0000 | 0.6667 | 0.8000 | 9 |
| **REFUTED** | 0.7500 | 1.0000 | 0.8571 | 6 |
| **UNCERTAIN** | 0.5000 | 0.6000 | 0.5455 | 5 |

### Confusion Matrix:
Labels: `['SUPPORTED', 'REFUTED', 'UNCERTAIN']`
```
[6, 0, 3]
[0, 6, 0]
[0, 2, 3]
```
