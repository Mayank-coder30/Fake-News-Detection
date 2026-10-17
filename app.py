"""EvidenceLens: Evidence-Grounded Fake News Detection and Fact Verification.
Interactive Streamlit Web Application.
Academic Project - NIT Delhi (CSBB 311)
Supervisor: Dr. Gautam Kumar
"""

import sys
from pathlib import Path
import json
import streamlit as st

# Configure page layout and metadata
st.set_page_config(
    page_title="EvidenceLens — Fact Verification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling and readability
st.markdown("""
<style>
    /* Main container styling */
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .badge-supported {
        background-color: #DEF7EC;
        color: #03543F;
        border: 1px solid #31C48D;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-refuted {
        background-color: #FDE8E8;
        color: #9B1C1C;
        border: 1px solid #F98080;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-uncertain {
        background-color: #FEF08A;
        color: #713F12;
        border: 1px solid #EAB308;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-error {
        background-color: #F3F4F6;
        color: #4B5563;
        border: 1px solid #9CA3AF;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-num {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .evidence-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin: 8px 0;
        border-radius: 0 6px 6px 0;
    }
    .explanation-box {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    REPORTS_DIR,
    LABEL_SUPPORTED,
    LABEL_REFUTED,
    LABEL_UNCERTAIN
)
from src.pipeline.fact_check_pipeline import FactCheckPipeline

# Cached pipeline initialization to load models once
@st.cache_resource(show_spinner="Loading NLP & Transformer Models (Sentence-Transformers + NLI Verifier)...")
def get_pipeline():
    return FactCheckPipeline()

# Sample articles for demonstration
SAMPLE_ARTICLES = {
    "Select a pre-loaded sample article...": "",
    "Sample 1: Verified Scientific Facts (Supported)": (
        "The James Webb Space Telescope (JWST) was launched on December 25, 2021 on an Ariane 5 rocket. "
        "The observatory is specifically designed to conduct infrared astronomy. "
        "Neil Armstrong and Buzz Aldrin landed on the lunar surface during the Apollo 11 mission on July 20, 1969."
    ),
    "Sample 2: Viral Sensational Claims (Refuted)": (
        "Viral social media posts claim that the Eiffel Tower was built entirely out of pure solid gold. "
        "Furthermore, conspiracy videos assert that Apollo 11 actually landed on Mars instead of the Moon. "
        "Historical records also claim Albert Einstein received the Nobel Prize in Literature."
    ),
    "Sample 3: Inconclusive / Missing Evidence (Uncertain)": (
        "Ancient scrolls supposedly indicate that Greek philosophers enjoyed strawberry gelato on hot summer afternoons. "
        "Neil Armstrong secretly listened to jazz clarinet solos while collecting lunar dust samples. "
        "A distant exoplanet contains microscopic glowing blue crystals that sing during midnight storms."
    ),
    "Sample 4: Mixed News Story (Multi-Verdict)": (
        "The James Webb Space Telescope was launched in December 2021 to observe distant galaxies. "
        "A recent viral headline claimed that Apollo 11 was launched to Mars in 1969. "
        "Secret government files allegedly confirm that Albert Einstein owned an orange tabby cat named Copernicus."
    )
}

def render_sidebar():
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/en/thumb/0/02/National_Institute_of_Technology%2C_Delhi_Logo.svg/1200px-National_Institute_of_Technology%2C_Delhi_Logo.svg.png", width=90)
        st.markdown("### **EvidenceLens**")
        st.caption("B.Tech CSE Project — CSBB 311\nNational Institute of Technology Delhi")

        st.markdown("---")
        st.markdown("#### **Project Team**")
        st.markdown("""
        - **Krrish Kumar** (241210070)
        - **Mayank** (241210075)
        - **Mishti Gupta** (241210079)
        - **Prakshi Agarwal** (241210091)
        - **Rachit Singh Rana** (241210097)
        """)
        st.markdown("**Supervisor**: Dr. Gautam Kumar")

        st.markdown("---")
        st.markdown("#### **System Architecture**")
        st.markdown("""
        1. **Claim Extraction**: NLP heuristic segmentation
        2. **Semantic Retrieval**: Sentence-Transformers + FAISS
        3. **Verification**: DistilRoBERTa Cross-Encoder NLI
        4. **Explanation**: Grounded Factual Reasoning
        """)

        st.markdown("---")
        st.caption("EvidenceLens v1.0.0 | Academic Research Prototype")

def render_dashboard(summary):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color: #1E293B;">{summary.total_claims}</div><div class="metric-label">Claims Extracted</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color: #057A55;">{summary.supported_count}</div><div class="metric-label">Supported</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color: #E02424;">{summary.refuted_count}</div><div class="metric-label">Refuted</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color: #D97706;">{summary.uncertain_count}</div><div class="metric-label">Uncertain</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color: #4B5563;">{summary.elapsed_seconds}s</div><div class="metric-label">Inference Time</div></div>', unsafe_allow_html=True)

def render_claim_card(claim, index):
    lbl = claim["label"]
    if lbl == LABEL_SUPPORTED:
        badge = f'<span class="badge-supported">✅ SUPPORTED (Confidence: {claim["confidence"]*100:.1f}%)</span>'
    elif lbl == LABEL_REFUTED:
        badge = f'<span class="badge-refuted">❌ REFUTED (Confidence: {claim["confidence"]*100:.1f}%)</span>'
    elif lbl == LABEL_UNCERTAIN:
        badge = f'<span class="badge-uncertain">⚠️ UNCERTAIN (Insufficient Evidence)</span>'
    else:
        badge = f'<span class="badge-error">⚙️ {lbl}</span>'

    with st.expander(f"**Claim {index}:** {claim['claim_text']}", expanded=True):
        st.markdown(f"**Verdict:** {badge}", unsafe_allow_html=True)
        st.markdown(f"**Claim ID:** `{claim['claim_id']}` | **Position in Article:** Sentence #{claim.get('position', index)}")

        st.markdown("#### **Evidence-Grounded Explanation**")
        st.info(f"{claim['explanation']}\n\n*Source: {claim.get('explanation_provider', 'Deterministic Grounded Engine')}*")

        st.markdown("#### **Retrieved Knowledge Evidence**")
        evidence_list = claim.get("evidence", [])
        if evidence_list:
            for p_idx, ev in enumerate(evidence_list, start=1):
                url_link = f" — [View Original Source]({ev['source_url']})" if ev.get("source_url") else ""
                st.markdown(f"""
                <div class="evidence-box">
                    <strong>Passage #{p_idx}: {ev.get('title', 'Unknown Source')}</strong> (Semantic Relevance: <code>{ev.get('relevance_score', 0.0):.4f}</code>){url_link}<br/>
                    <em>"{ev.get('text', '')}"</em>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No relevant evidence passages met the minimum relevance threshold in the knowledge corpus.")

def main():
    render_sidebar()

    st.markdown('<div class="main-title">🔍 EvidenceLens</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Evidence-Grounded Fake News Detection & Fact Verification System</div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "🚀 Fact Check Article",
        "📊 Benchmark & Comparison",
        "🧠 How It Works",
        "ℹ️ System Information",
        "⚠️ Limitations & Ethics"
    ])

    with tabs[0]:
        st.markdown("#### Enter News Article or Headline for Verification")
        st.caption("Paste any news text below. EvidenceLens will decompose the article into factual claims, retrieve evidence from the indexed corpus, and verify each claim transparently.")

        sample_choice = st.selectbox("Or choose a pre-loaded demonstration article:", list(SAMPLE_ARTICLES.keys()))
        default_input = SAMPLE_ARTICLES[sample_choice] if sample_choice else ""

        article_input = st.text_area(
            "Article Text:",
            value=default_input,
            height=150,
            placeholder="e.g. The James Webb Space Telescope conducts infrared astronomy. Scientists confirmed new observations..."
        )

        col_btn1, col_btn2, _ = st.columns([1.5, 1.2, 5])
        with col_btn1:
            analyze_clicked = st.button("🔎 Analyze Article", type="primary", use_container_width=True)
        with col_btn2:
            clear_clicked = st.button("🗑️ Clear Input", use_container_width=True)

        if clear_clicked:
            st.rerun()

        if analyze_clicked:
            if not article_input.strip():
                st.error("Please provide non-empty article text to verify.")
            else:
                with st.spinner("Processing article: Segmenting claims, searching FAISS index, and evaluating NLI cross-encoder..."):
                    pipeline = get_pipeline()
                    result = pipeline.analyze_article(article_input)

                st.markdown("---")
                st.markdown("### 📋 Verification Summary")
                render_dashboard(result.summary)
                st.caption(f"ℹ️ {result.summary.disclaimer}")

                st.markdown("---")
                st.markdown("### 🔎 Claim-by-Claim Evidence Breakdown")
                if not result.claims:
                    st.info("No verifiable factual declarative claims were detected in the supplied text.")
                else:
                    for idx, clm in enumerate(result.claims, start=1):
                        render_claim_card(clm, idx)

    with tabs[1]:
        st.markdown("### 📈 Experimental Benchmark & Model Comparison")
        st.markdown("""
        In accordance with the project synopsis, this system contrasts the proposed **Evidence-Grounded Verification Pipeline**
        against a **Conventional Text Classification Baseline** (TF-IDF + Logistic Regression).
        """)

        # Load metrics
        pipeline_metrics_file = REPORTS_DIR / "pipeline_metrics.json"
        baseline_metrics_file = REPORTS_DIR / "baseline_metrics.json"

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("#### 1. Conventional Text Classifier (Baseline)")
            if baseline_metrics_file.exists():
                with open(baseline_metrics_file, "r", encoding="utf-8") as f:
                    b_data = json.load(f)
                st.markdown(f"""
                - **Model Architecture**: TF-IDF (1-2 n-grams) + Logistic Regression
                - **Task Formulation**: Coarse Article-level Binary Classification (`REAL` vs `FAKE`)
                - **Accuracy**: `{b_data.get('accuracy', 0.0):.4f}`
                - **Macro F1-Score**: `{b_data.get('macro_f1', 0.0):.4f}`
                - **Evaluated Test Samples**: `{b_data.get('sample_count', 0)}`
                """)
                st.caption("⚠️ Limitation: Relies solely on surface stylistic cues; does not verify external factual truth.")
            else:
                st.warning("Baseline evaluation metrics report not found. Run `scripts/train_baseline.py`.")

        with col_b2:
            st.markdown("#### 2. Evidence-Grounded Pipeline (Proposed)")
            if pipeline_metrics_file.exists():
                with open(pipeline_metrics_file, "r", encoding="utf-8") as f:
                    p_data = json.load(f)
                p_cls = p_data.get("classification_metrics", {})
                p_ret = p_data.get("retrieval_metrics", {})
                st.markdown(f"""
                - **Model Architecture**: SentenceTransformer (`all-MiniLM-L6-v2`) + FAISS + Transformer NLI (`nli-distilroberta-base`)
                - **Task Formulation**: Claim-level Verification (`SUPPORTED`, `REFUTED`, `UNCERTAIN`)
                - **Overall Accuracy**: `{p_cls.get('accuracy', 0.0):.4f}`
                - **Macro F1-Score**: `{p_cls.get('macro_f1', 0.0):.4f}`
                - **Evidence Recall@k**: `{p_ret.get('recall_at_top_k', 0.0):.4f}`
                - **Mean Reciprocal Rank (MRR)**: `{p_ret.get('mrr', 0.0):.4f}`
                - **Evaluated Benchmark Records**: `{p_data.get('total_samples', 0)}` (FEVER Benchmark)
                """)
                st.caption("✅ Advantage: Provides verifiable evidence citations and epistemic uncertainty handling.")
            else:
                st.warning("Pipeline benchmark metrics report not found. Run `scripts/run_evaluation.py`.")

        st.markdown("---")
        st.markdown("#### Per-Class Verification Performance (FEVER Benchmark)")
        if pipeline_metrics_file.exists():
            per_cls = p_data.get("classification_metrics", {}).get("per_class", {})
            st.table({
                "Category": list(per_cls.keys()),
                "Precision": [f"{v['precision']:.4f}" for v in per_cls.values()],
                "Recall": [f"{v['recall']:.4f}" for v in per_cls.values()],
                "F1-Score": [f"{v['f1_score']:.4f}" for v in per_cls.values()],
                "Support (Claims)": [v["support"] for v in per_cls.values()]
            })

    with tabs[2]:
        st.markdown("### 🧠 How EvidenceLens Works")
        st.markdown("""
        EvidenceLens breaks away from black-box fake news classification by dividing verification into four auditable stages:

        ```mermaid
        graph LR
            A[Raw Article Text] --> B[Claim Extractor]
            B --> C[Candidate Claims]
            C --> D[Embedding Model]
            D --> E[FAISS Vector Index]
            E --> F[Top-k Ranked Evidence]
            F --> G[Transformer NLI Verifier]
            G --> H{Verdict}
            H -->|Evidence Entails| I[SUPPORTED]
            H -->|Evidence Contradicts| J[REFUTED]
            H -->|Evidence Missing| K[UNCERTAIN]
            I & J & K --> L[Evidence-Grounded Explainer]
        ```

        1. **Text Preprocessing & Claim Extraction**:
           The input article is normalized and segmented into individual declarative sentences. Non-verifiable sentences (such as questions, conversational remarks, or subjective opinions) are filtered out, preserving candidate factual claims.
        2. **Semantic Evidence Retrieval**:
           Each claim is mapped into a 384-dimensional dense semantic space via `all-MiniLM-L6-v2`. FAISS searches the indexed corpus using inner product similarity (equivalent to cosine similarity on normalized vectors), retrieving the top candidate passages.
        3. **Transformer-Based Claim Verification**:
           Retrieved evidence passages and the claim are fed into a pretrained NLI Cross-Encoder (`cross-encoder/nli-distilroberta-base`). The model evaluates the relationship:
           - **SUPPORTED**: The evidence explicitly entails the claim.
           - **REFUTED**: The evidence directly contradicts the claim.
           - **UNCERTAIN**: Relevant evidence is absent, inconclusive, or conflicting.
        4. **Transparent Explanation Generation**:
           The system generates an interpretable justification quoting the verified source passages and identifying specific reasoning.
        """)

    with tabs[3]:
        st.markdown("### ℹ️ System & Model Information")
        st.markdown("""
        - **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
        - **Verification Model**: `cross-encoder/nli-distilroberta-base` (DistilRoBERTa sequence classifier)
        - **Vector Index Engine**: `FAISS-CPU` (`IndexFlatIP` inner-product search)
        - **Primary Benchmark Dataset**: FEVER (Fact Extraction and VERification)
        - **Conventional Baseline**: Scikit-Learn TF-IDF Vectorizer + Logistic Regression
        - **Explanation Provider**: Deterministic Rule-Grounded Engine (with optional OpenAI/Groq/Gemini API integration)
        - **Execution Hardware**: Intel CPU / AMD Ryzen (Compatible with Local CPU and CUDA GPU)
        """)

    with tabs[4]:
        st.markdown("### ⚠️ Limitations & Ethical Considerations")
        st.markdown("""
        1. **Corpus Coverage**:
           The verification pipeline can only verify claims for which evidence exists in the indexed knowledge source. If an event occurred after the knowledge base was indexed, the system will correctly classify the claim as **UNCERTAIN** rather than false.
        2. **Absence of Evidence is Not Evidence of Absence**:
           EvidenceLens explicitly distinguishes between a claim being refuted versus insufficient evidence being found. An UNCERTAIN label must not be interpreted as confirmation that a claim is fraudulent.
        3. **Subjective vs. Factual Nuance**:
           Claim extraction filters obvious opinions, but nuanced figurative language, sarcasm, or value judgments may require human journalistic editorial oversight.
        4. **Research Prototype Notice**:
           This software is developed for academic research at NIT Delhi to advance interpretable misinformation detection.
        """)

if __name__ == "__main__":
    main()
