# EvidenceLens Performance Metrics

**Project:** Evidence-Grounded Fake News Detection and Fact Verification  
**Generated:** 2026-11-04  
**Evaluation commands:**

```powershell
.\.venv310\Scripts\python.exe scripts\train_baseline.py
.\.venv310\Scripts\python.exe scripts\run_evaluation.py
```

## 1. Evaluation Scope

| Evaluation | Dataset | Split | Samples | Classes |
|---|---|---:|---:|---|
| Article baseline | `data/sample/baseline_fake_news.jsonl` | 70% train / 30% test | 101 / 31 test | `FAKE`, `REAL` |
| FEVER baseline | `data/processed/fever_processed.jsonl` | 70% train / 30% test | 500 / 150 test | `SUPPORTED`, `REFUTED`, `UNCERTAIN` |
| EvidenceLens pipeline | `data/processed/fever_processed.jsonl` | Full evaluation file | 500 | `SUPPORTED`, `REFUTED`, `UNCERTAIN` |

The article and FEVER baselines use TF-IDF features with Logistic Regression. The
EvidenceLens pipeline uses dense embedding retrieval, a 100-passage FAISS index,
and Transformer NLI verification with `top_k = 3`.

## 2. Aggregate Comparison

| Metric | Article baseline | FEVER baseline | EvidenceLens pipeline |
|---|---:|---:|---:|
| Accuracy | **0.9677 (96.77%)** | **0.3200 (32.00%)** | **0.3280 (32.80%)** |
| Error rate | 0.0323 (3.23%) | 0.6800 (68.00%) | 0.6720 (67.20%) |
| Correct predictions | 30 / 31 | 48 / 150 | 164 / 500 |
| Macro precision | 0.9706 | 0.3268 | 0.2267 |
| Macro recall / balanced accuracy | 0.9667 | 0.3200 | 0.3293 |
| Macro F1-score | **0.9676** | **0.3181** | **0.1842** |

The pipeline result above is the current full 500-record evaluation. It replaces
the older 60-record result that had previously been displayed in the application.

## 3. Article-Level Baseline

**Model:** TF-IDF + Logistic Regression  
**Dataset:** 101 news articles, stratified 70/30 split, 31 test articles

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `FAKE` | 1.0000 | 0.9333 | 0.9655 | 15 |
| `REAL` | 0.9412 | 1.0000 | 0.9697 | 16 |
| **Macro average** | **0.9706** | **0.9667** | **0.9676** | 31 |

### Confusion Matrix

Labels: `FAKE`, `REAL`

```text
[[14, 1],
 [ 0,16]]
```

This means 14 fake articles and all 16 real articles were classified correctly.
One fake article was classified as real.

## 4. FEVER Claim-Level Baseline

**Model:** TF-IDF + Logistic Regression  
**Dataset:** 500 FEVER claims, stratified 70/30 split, 150 test claims

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `REFUTED` | 0.3529 | 0.2400 | 0.2857 | 50 |
| `SUPPORTED` | 0.2812 | 0.3600 | 0.3158 | 50 |
| `UNCERTAIN` | 0.3462 | 0.3600 | 0.3529 | 50 |
| **Macro average** | **0.3268** | **0.3200** | **0.3181** | 150 |

### Confusion Matrix

Labels: `REFUTED`, `SUPPORTED`, `UNCERTAIN`

```text
[[12, 24, 14],
 [12, 18, 20],
 [10, 22, 18]]
```

## 5. EvidenceLens Pipeline

**Model:** Sentence-Transformers + FAISS + Transformer NLI  
**Dataset:** 500 FEVER claims  
**Evidence index:** 100 passages, 384-dimensional vectors  
**Retrieval setting:** top 3 passages, relevance threshold 0.35

### Classification Metrics

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `SUPPORTED` | 0.0000 | 0.0000 | 0.0000 | 168 |
| `REFUTED` | 0.3529 | 0.0361 | 0.0656 | 166 |
| `UNCERTAIN` | 0.3271 | 0.9518 | 0.4869 | 166 |
| **Macro average** | **0.2267** | **0.3293** | **0.1842** | 500 |

### Confusion Matrix

Labels: `SUPPORTED`, `REFUTED`, `UNCERTAIN`

```text
[[  0,  3, 165],
 [  0,  6, 160],
 [  0,  8, 158]]
```

### Retrieval Metrics

| Metric | Value |
|---|---:|
| Claims with annotated evidence | 19 |
| Retrieval hits in top 3 | 0 |
| Retrieval misses in top 3 | 19 |
| Recall@3 | **0.0000** |
| Mean Reciprocal Rank (MRR) | **0.0000** |

The pipeline predicted `UNCERTAIN` for most claims because no indexed passage
matched the annotated evidence references in this full evaluation file. This
retrieval failure explains the low supported-class recall and should be treated
as a current system limitation, not as evidence that the underlying approach is
incorrect.

## 6. Metric Interpretation

- **Accuracy** is the fraction of all predictions that match the gold label.
- **Precision** measures how often predictions for a class are correct.
- **Recall** measures how many gold examples of a class are found.
- **F1-score** is the harmonic mean of precision and recall.
- **Macro averages** give each class equal weight.
- **Recall@3** measures whether at least one annotated evidence document appears
  among the three retrieved passages.
- **MRR** rewards retrieving the first relevant passage at a high rank.

## 7. Metrics Not Reported

ROC-AUC, PR-AUC, calibration error, latency percentiles, memory usage, and
confidence intervals are not calculated by the repository's current evaluation
implementation. The saved reports expose class predictions and confidence
values, but they do not define a thresholding protocol for these additional
metrics. They should not be inferred from the aggregate results above.

## 8. Source Reports

- [`reports/baseline_metrics.json`](reports/baseline_metrics.json)
- [`reports/fever_baseline_metrics.json`](reports/fever_baseline_metrics.json)
- [`reports/pipeline_metrics.json`](reports/pipeline_metrics.json)
- [`reports/model_comparison.md`](reports/model_comparison.md)
