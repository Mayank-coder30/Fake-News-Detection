# Comparative Evaluation Report: Conventional Baseline vs. Evidence-Grounded Pipeline
**Project**: Evidence-Grounded Fake News Detection and Fact Verification (EvidenceLens)  
**Academic Institution**: National Institute of Technology Delhi (CSBB 311)  
**Supervisor**: Dr. Gautam Kumar  

---

## 1. Executive Summary

This benchmark rigorously evaluates both paradigms on the **same claim verification dataset**:
1. **Conventional Surface-Level Baseline**: Uses lexical n-grams (TF-IDF + Logistic Regression) without external knowledge retrieval.
2. **EvidenceLens Proposed Pipeline**: Decomposes propositions, retrieves evidence from an indexed knowledge corpus via dense semantic embeddings (Sentence-Transformers + FAISS), and conducts Transformer Cross-Encoder NLI inference to classify claims into **SUPPORTED**, **REFUTED**, or **UNCERTAIN**.

---

## 2. Quantitative Performance Comparison on the Same Dataset (N=60)

| Metric / Dimension | Conventional Baseline (Claims) | EvidenceLens Proposed Pipeline | Relative Advantage |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | TF-IDF (1-2 n-grams) + Logistic Regression | Dense Embeddings + FAISS + DistilRoBERTa NLI | Deep Cross-Attention |
| **Input Unit** | Factual Claim Sentences | Factual Claim Sentences | Identical |
| **Knowledge Grounding** | None (Closed surface words) | Open-world Retrieval from 100 Verified Passages | +100% Evidence Grounding |
| **Target Taxonomy** | 3-Class (`SUPPORTED`, `REFUTED`, `UNCERTAIN`) | 3-Class (`SUPPORTED`, `REFUTED`, `UNCERTAIN`) | Identical |
| **Evaluation Dataset** | FEVER Aligned Benchmark (N=60) | FEVER Aligned Benchmark (N=60) | **Same Dataset** |
| **Overall Accuracy** | **31.67%** | **91.67%** | **+60.00% absolute gain** |
| **Macro F1-Score** | **0.3152** | **0.9126** | **+0.5974 gain** |
| **Evidence Recall@k** | N/A (No retrieval) | **1.0000 (100%)** | Full retrieval coverage |
| **Retrieval MRR** | N/A (No retrieval) | **1.0000** | Perfect top rank |
| **Auditability** | 0% (Word frequencies) | **100%** (Cites exact text & similarity) | Transparent audit trail |
| **Epistemic Modality** | None (Forced guess) | **Yes** (Safely predicts `UNCERTAIN`) | Prevents hallucinations |

---

## 3. Scientific Discussion & Key Findings

### Why the Baseline Fails (31.67% Accuracy)
When evaluated on factual claim verification, the conventional TF-IDF baseline achieves only **31.67% accuracy** (indistinguishable from random guessing across 3 balanced classes).
This demonstrates a core scientific principle: **the veracity of a factual claim cannot be determined from word frequencies alone**.

For example:
- *"Albert Einstein was awarded the Nobel Prize in Physics."* (SUPPORTED)
- *"Albert Einstein was awarded the Nobel Prize in Literature."* (REFUTED)

Both statements share identical vocabulary, grammar, and formal tone. Without retrieving external evidence, a bag-of-words classifier is unable to distinguish truth from falsehood.

### Why EvidenceLens Excels (91.67% Accuracy)
By grounding claims in empirical evidence via FAISS dense retrieval and cross-encoder NLI:
1. The pipeline retrieves the authentic reference passage with 100% recall.
2. The cross-encoder model measures full cross-attention between the claim and the retrieved passage, identifying subtle factual contradictions.
3. When evidence is genuinely missing, the system outputs `UNCERTAIN`, ensuring epistemic safety.

---

## 4. Per-Class Performance Breakdown (EvidenceLens)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **SUPPORTED** | 0.9600 | 0.9231 | 0.9412 | 26 |
| **REFUTED** | 0.8571 | 0.9474 | 0.9000 | 19 |
| **UNCERTAIN** | 0.9286 | 0.8667 | 0.8966 | 15 |

### Confusion Matrix:
Labels: `['SUPPORTED', 'REFUTED', 'UNCERTAIN']`
```
[24,  1,  1]   # SUPPORTED (24 correct)
[ 1, 18,  0]   # REFUTED (18 correct)
[ 0,  2, 13]   # UNCERTAIN (13 correct)
```
