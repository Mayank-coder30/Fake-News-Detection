# Academic Viva & Defense Preparation Guide
**Project**: Evidence-Grounded Fake News Detection and Fact Verification (EvidenceLens)  
**Institution**: National Institute of Technology Delhi | Department of Computer Science & Engineering (Semester 5)  
**Supervisor**: Dr. Gautam Kumar  
**Students**: Krrish Kumar, Mayank, Mishti Gupta, Prakshi Agarwal, Rachit Singh Rana

---

### Q1: What core problem does this project solve?
**Answer**:  
Digital news and social media allow fabricated stories to spread rapidly. Traditional fake news detectors treat misinformation detection as a simple binary text-classification task (predicting Real or Fake from writing style). However, stylistic patterns do not verify whether factual claims are actually true. This project solves that problem by building an **evidence-grounded fact verification system** that breaks down articles into individual claims, retrieves verifiable evidence from a knowledge corpus, determines whether that evidence supports or refutes each claim, and generates a transparent explanation.

---

### Q2: Why is conventional fake-news classification insufficient?
**Answer**:  
1. **Superficial Stylistic Bias**: Conventional models (like TF-IDF + classifiers or fine-tuned BERT classifiers) rely on stylistic proxies such as sensationalism or vocabulary cues. Well-written, subtle fake news easily tricks them.
2. **Lack of Auditability**: A binary "Fake" label offers zero proof or evidence to convince users or journalists.
3. **Inability to Express Uncertainty**: Real articles frequently contain mixed statements—some true, some false, and some unverified. A binary label oversimplifies reality.
4. **Distribution Shift**: Surface classifiers degrade severely when encountering new topics or breaking events not present in the training set.

---

### Q3: What is evidence-grounded fact verification?
**Answer**:  
Evidence-grounded fact verification is a knowledge-driven NLP paradigm where the veracity of a statement is determined strictly by its logical relationship (entailment, contradiction, or neutral) with retrieved, authoritative evidence passages. The prediction is tied directly to verifiable citations.

---

### Q4: What is the FEVER dataset?
**Answer**:  
FEVER stands for **Fact Extraction and VERification**, introduced by Thorne et al. (NAACL-HLT 2018). It is the premier research benchmark consisting of 185,445 claims generated from Wikipedia and verified against Wikipedia introductory pages. Each claim is labeled as `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO`.

---

### Q5: What is a claim vs an evidence passage?
**Answer**:  
- **Claim**: A discrete declarative statement asserting a checkable fact (e.g., *"The James Webb Space Telescope conducts infrared astronomy"*).
- **Evidence Passage**: A retrieved contextual excerpt from a verified source document (such as an encyclopedia or peer-reviewed document) containing relevant factual ground truth.

---

### Q6: Why use Sentence Transformers instead of standard TF-IDF for retrieval?
**Answer**:  
TF-IDF relies on exact keyword matching and fails when a claim uses synonyms, paraphrasing, or altered sentence structure. Sentence Transformers (specifically `all-MiniLM-L6-v2`) map sentences into a dense 384-dimensional continuous semantic space where conceptually similar statements lie close to each other, enabling semantic search even with zero lexical overlap.

---

### Q7: What is FAISS and why is it used?
**Answer**:  
FAISS (**Facebook AI Similarity Search**) is an open-source library optimized for high-speed similarity search across high-dimensional vector spaces. In EvidenceLens, we use FAISS's `IndexFlatIP` (Inner Product on unit-normalized vectors). This provides exact Cosine Similarity at millisecond scale on CPU.

---

### Q8: What transformer architecture was used for verification?
**Answer**:  
We utilize `cross-encoder/nli-distilroberta-base`, a DistilRoBERTa model fine-tuned for Natural Language Inference (NLI). In contrast to bi-encoders that process sentences separately, a cross-encoder performs full cross-attention over the concatenated `[CLS] Evidence [SEP] Claim [SEP]` tokens, allowing full contextual interaction between every word in the claim and every word in the evidence.

---

### Q9: What is the difference between SUPPORTED, REFUTED, and UNCERTAIN?
**Answer**:  
- **SUPPORTED**: The retrieved evidence explicitly entails and confirms the factual assertion of the claim.
- **REFUTED**: The retrieved evidence directly contradicts and disproves the claim.
- **UNCERTAIN**: Relevant evidence is absent from the corpus, inconclusive, or conflicting across reputable sources.

---

### Q10: How does the system handle missing evidence?
**Answer**:  
**Critical rule**: *The absence of evidence is not evidence of falsehood*. If the semantic retriever cannot locate passages with a relevance score above the threshold ($0.35$), the system automatically assigns `UNCERTAIN` and produces an explanation stating that sufficient evidence is currently unavailable in the knowledge corpus. It never falsely marks a claim as refuted simply because evidence was not found.

---

### Q11: What is the purpose of an LLM in this system?
**Answer**:  
The LLM serves exclusively as an **explainer**, not as an arbitrary judge of truth. It translates the mathematical verification outcome and the retrieved evidence passage into a clear, concise natural language justification. If no external LLM API key is configured, our deterministic rule engine synthesizes the explanation directly from the evidence text.

---

### Q12: How do you prevent LLM hallucinations?
**Answer**:  
1. The classification label is decided by the transformer verifier, **not** the LLM. The LLM cannot change the label.
2. The prompt restricts reasoning strictly to the provided evidence passage.
3. Temperature is set low ($0.2$) to minimize creative deviation.
4. A deterministic, non-generative fallback is active by default.

---

### Q13: How was the baseline classifier implemented and compared?
**Answer**:  
We implemented a conventional text classifier using TF-IDF (1-2 n-grams) and Logistic Regression trained on article-level Real and Fake news. On the held-out test split, the baseline achieved an accuracy of 62.50% and Macro-F1 of 0.5636, but provided zero evidence or transparency. In contrast, the evidence-grounded pipeline achieved 75.00% accuracy and 0.7342 Macro-F1 on the FEVER benchmark, with 100% evidence citation coverage.

---

### Q14: How is Accuracy and Macro-F1 calculated?
**Answer**:  
- **Accuracy**: The total number of correct predictions divided by the total number of evaluated samples.
- **Macro-F1**: The unweighted arithmetic average of the F1 scores across all three classes:
  $$\text{Macro-F1} = \frac{F1_{\text{SUPPORTED}} + F1_{\text{REFUTED}} + F1_{\text{UNCERTAIN}}}{3}$$
  Macro-F1 is essential for multiclass evaluation because it penalizes models that perform poorly on minority classes.

---

### Q15: What did the team implement vs what was reused from pretrained models?
**Answer**:  
- **Pretrained Components Reused**:
  - `sentence-transformers/all-MiniLM-L6-v2` for dense sentence embeddings.
  - `cross-encoder/nli-distilroberta-base` for natural language inference.
- **Original Implementation Built by the Team**:
  - End-to-end modular pipeline orchestration (`src/pipeline/`).
  - Rule-based factual claim extraction and sentence filtering (`src/claim_extraction/`).
  - FAISS index construction, persistence, and similarity thresholding (`src/evidence_retrieval/`).
  - Relevance-stratified multi-evidence aggregation algorithm (`src/verification/`).
  - Deterministic grounded explanation engine and anti-hallucination prompts (`src/explanation/`).
  - Conventional text-classification baseline and comparative evaluation suite (`src/evaluation/`).
  - Full interactive Streamlit web dashboard (`app.py`).
  - 26 unit and integration tests (`tests/`).
