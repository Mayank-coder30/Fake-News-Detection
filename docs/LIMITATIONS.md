# System Limitations and Engineering Boundaries

## 1. Knowledge Base Coverage & The Closed-World Assumption
- **Static Corpus Constraint**: The semantic retriever searches an indexed corpus. If an event occurred after corpus indexing or relates to local, unindexed subjects, no matching evidence will be retrieved.
- **Handling**: EvidenceLens explicitly flags such claims as `UNCERTAIN` rather than fabricating false refutations. Users must recognize that `UNCERTAIN` reflects corpus boundaries, not verified falsity.

---

## 2. Temporal Obsolescence
- Political offices, scientific discoveries, sports records, and policy decisions evolve dynamically over time.
- A passage valid in 2018 (e.g. stating that a leader is currently in office) may conflict with current reality.
- Future iterations can integrate temporal timestamps on evidence passages to perform time-aware reasoning.

---

## 3. Claim Extraction Nuances
- Heuristic claim extraction relies on declarative syntactic structures, entities, and factual markers.
- **False Negatives**: Highly subtle factual claims phrased within conditional clauses or rhetorical assertions may be filtered out.
- **False Positives**: Value judgments masquerading as assertions ("The policy was a disastrous failure") may require deeper semantic parsing or full dependency parsing.

---

## 4. Probabilistic Calibration vs Truth Confidence
- Softmax output probabilities produced by transformer heads indicate model confidence within the learned distribution, **not** real-world ontological truth probabilities.
- A 90% softmax confidence means the model found strong token-level alignment between evidence and hypothesis, but should not replace human editorial scrutiny.

---

## 5. Hallucination Risks with Generative LLMs
- When optional generative LLMs (e.g. OpenAI or Groq) are enabled to formulate natural-language explanations, there is a known risk of generative embellishment.
- **Safeguards Implemented**:
  1. Temperature is locked low ($0.2$).
  2. Prompts enforce strict grounding exclusively within the retrieved passages.
  3. The LLM is strictly prohibited from altering the classifier's verdict.
  4. A deterministic, non-generative rule engine is provided as the default fallback.

---

## 6. Real-World Web Retrieval Trade-offs
- Searching the live open web introduces noise, paywalled articles, SEO manipulation, and conflicting blog posts.
- For academic benchmarking and reproducible evaluation, a curated, authenticated corpus (like FEVER) provides verifiable ground truth.
