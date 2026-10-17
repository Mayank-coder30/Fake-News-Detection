# Technical Methodology: Evidence-Grounded Verification

## 1. Problem Formulation
Let an input article $A$ be represented as a sequence of natural language sentences:
$$A = (s_1, s_2, \dots, s_n)$$

Traditional fake news classification attempts to learn a direct mapping:
$$f(A) \rightarrow \{0, 1\} \quad (\text{Fake}, \text{Real})$$

This formulation suffers from several fundamental weaknesses:
1. **Confounding Stylistic Bias**: The model captures surface lexical cues (e.g. clickbait syntax) rather than factual veracity.
2. **Lack of Epistemic Modality**: Unable to express that evidence is missing or inconclusive.
3. **No Explainability**: Offers no factual justification or citation to refute misinformation.

### Proposed Evidence-Grounded Paradigm
In EvidenceLens, the task is reformulated into a 4-step pipeline:
1. **Claim Extraction**: $E(A) \rightarrow \mathcal{C} = \{c_1, c_2, \dots, c_m\}$ where each $c_i$ is an independently checkable declarative proposition.
2. **Dense Evidence Retrieval**: For each claim $c$, retrieve top-$k$ passages $\mathcal{P}_c = \{p_1, \dots, p_k\}$ from knowledge base $\mathcal{K}$ maximizing semantic similarity:
   $$\text{sim}(c, p) = \frac{\mathbf{e}_c \cdot \mathbf{e}_p}{\|\mathbf{e}_c\|_2 \|\mathbf{e}_p\|_2}$$
3. **Claim-Evidence Verification**: A transformer-based Natural Language Inference (NLI) model scores the relationship:
   $$\mathcal{M}(p, c) \rightarrow [P(\text{Support}), P(\text{Refute}), P(\text{Uncertain})]$$
4. **Grounded Explanation Generation**: An explanatory module generates transparent justification anchored directly in $\mathcal{P}_c$.

---

## 2. Claim Extraction Methodology
News articles contain mixed discourse modes: opinions, rhetorical questions, subjective analysis, and factual assertions. The claim extractor isolates verifiable declarative statements via heuristic linguistic filtering:
- **Filtering Rules**:
  - Exclude interrogative sentences ($s$ ending in `?`).
  - Exclude subjective opinion discourse markers (`"In my opinion"`, `"I believe"`, `"Personally"`).
  - Exclude conversational greetings and call-to-actions.
  - Require minimum length ($\ge 4$ words) and predicate existence.
- **Factual Scoring Heuristic**: Increases extraction confidence when sentences contain dates, numbers, named entities, currency values, or factual predicates (`"announced"`, `"discovered"`, `"passed"`).

---

## 3. Dense Retrieval & FAISS Vector Search
- **Encoder**: We employ `all-MiniLM-L6-v2`, a 6-layer MiniLM model fine-tuned on >1 billion sentence pairs for semantic search.
- **Dimensionality**: Each passage and claim is projected into $\mathbb{R}^{384}$.
- **Inner Product & Cosine Equivalence**: By applying unit $L_2$ normalization during encoding ($\|\mathbf{e}\| = 1$), the inner product calculated by FAISS's `IndexFlatIP` is mathematically identical to cosine similarity:
  $$\langle \hat{\mathbf{e}}_c, \hat{\mathbf{e}}_p \rangle = \cos(\mathbf{e}_c, \mathbf{e}_p)$$
- **Thresholding**: Passages with similarity score $< 0.35$ are discarded to prevent out-of-domain false retrievals.

---

## 4. Transformer-Based Claim Verification
The verifier evaluates pairs using `cross-encoder/nli-distilroberta-base`. Unlike bi-encoders that encode sequences independently, cross-encoders perform full bidirectional cross-attention over all tokens in the evidence-claim pair simultaneously:
$$\mathbf{X} = [\text{CLS}] \circ p \circ [\text{SEP}] \circ c \circ [\text{SEP}]$$

The classification head outputs logits over three canonical categories:
- **SUPPORTED** ($y = 1$): Evidence entails the claim.
- **REFUTED** ($y = 0$): Evidence contradicts the claim.
- **UNCERTAIN** ($y = 2$): Evidence is neutral or insufficient.

### Multi-Evidence Aggregation
When multiple passages are retrieved for a claim, EvidenceLens employs relevance-stratified aggregation:
1. If the top-1 ranked passage exhibits high semantic similarity ($\ge 0.60$), it serves as the primary factual anchor.
2. Contradiction takes precedence if high-confidence refuting evidence ($\ge 0.60$) is identified.
3. If both strong support and strong refutation exist across reputable passages, the claim is assigned `UNCERTAIN` with an explicit `conflict_detected` warning.
4. If no passage meets the relevance threshold, the pipeline automatically assigns `UNCERTAIN`, ensuring missing evidence is never misinterpreted as evidence of falsehood.

---

## 5. Grounded Explanation Generation
To prevent hallucination:
- In deterministic mode, the explanation extracts genuine quotations, document titles, and similarity scores directly from retrieved evidence.
- In optional LLM mode, the prompt is tightly constrained to the retrieved context; system rules strictly prohibit the LLM from altering the classifier's verdict or adding external unverifiable assertions.
