# Dataset Documentation: FEVER Benchmark & Evidence Corpus

## 1. Overview
The primary benchmark utilized for training, validation, and evaluation in this project is the **FEVER (Fact Extraction and VERification)** dataset.

- **Official Project Page**: [https://fever.ai/](https://fever.ai/)
- **Authors**: James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Arpit Mittal (NAACL-HLT 2018)
- **Primary Domain**: Factual claims extracted and validated against English Wikipedia introductory articles.

---

## 2. Taxonomy & Label Mapping

FEVER defines a 3-class classification problem. EvidenceLens standardizes these labels into its canonical taxonomy:

| FEVER Official Label | EvidenceLens Canonical Label | Semantic Meaning |
| :--- | :--- | :--- |
| `SUPPORTS` | `SUPPORTED` | The available evidence contains sufficient factual information to prove the claim is true. |
| `REFUTES` | `REFUTED` | The available evidence contains sufficient factual information to prove the claim is false. |
| `NOT ENOUGH INFO` | `UNCERTAIN` | The available evidence is insufficient to confirm or deny the factual claim. |

---

## 3. Dataset Formats

### 3.1 Raw FEVER Format
Raw FEVER data is distributed as JSON Lines (`.jsonl`), where each entry includes:
- `id`: Integer claim identifier.
- `claim`: Natural language claim string.
- `label`: `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO`.
- `evidence`: Nested list of evidence sets: `[annotation_id, evidence_id, wikipedia_page_title, line_number]`.

### 3.2 Processed Format (`data/processed/fever_processed.jsonl`)
Standardized by `scripts/prepare_dataset.py`:
```json
{
  "id": 1001,
  "claim": "The James Webb Space Telescope conducts infrared astronomy.",
  "label": "SUPPORTED",
  "original_label": "SUPPORTS",
  "evidence_count": 1,
  "evidence_refs": [
    {
      "doc_title": "doc_001",
      "line_number": null
    }
  ]
}
```

### 3.3 Knowledge Evidence Corpus (`data/sample/evidence_corpus.jsonl`)
The indexed retrieval collection comprises verified Wikipedia reference passages:
```json
{
  "passage_id": "doc_001",
  "title": "James Webb Space Telescope",
  "text": "The James Webb Space Telescope (JWST) is a space telescope specifically designed to conduct infrared astronomy...",
  "source_id": "wiki_jwst",
  "source_url": "https://en.wikipedia.org/wiki/James_Webb_Space_Telescope"
}
```

---

## 4. Conventional Baseline Dataset (`data/sample/baseline_fake_news.jsonl`)
To compare against conventional text-only fake news detection (as required by the synopsis), a curated article-level dataset is provided with binary `REAL` and `FAKE` labels.
- Evaluated on a 70% / 30% stratified train-test split.
- Features articles from credible scientific/governmental press releases versus fabricated viral hoaxes.

---

## 5. Licensing & Attribution
- FEVER is licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0) license.
- Wikipedia text is distributed under Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
- Datasets are intended strictly for educational and academic research evaluation.
