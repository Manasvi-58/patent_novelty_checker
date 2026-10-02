import streamlit as st
import pandas as pd
import numpy as np
from preprocessing import clean_text, build_tfidf_matrix
from model import compute_novelty_score

# Page Configuration
st.set_page_config(
    page_title="Patent Novelty Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Enhancement
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .main {
        padding: 2rem;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header Styling */
    .title-text {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle-text {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    
    /* Section Cards */
    .css-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
    }
    
    /* Result Cards */
    .metric-container {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
        margin-top: 0.5rem;
    }
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        opacity: 0.8;
    }
    </style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="title-text">🔍 Patent Novelty Checker</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Evaluate invention novelty using hybrid TF-IDF keyword vectorization & Sentence-Transformer semantic search.</div>', unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Fast load using pre-computed embeddings and TF-IDF matrix."""
    df = pd.read_csv("dataset/patent_dataset.csv")
    dense_embeddings = np.load("dataset/embeddings.npy")
    tfidf_vectorizer, tfidf_matrix = build_tfidf_matrix(df)
    return df, tfidf_vectorizer, tfidf_matrix, dense_embeddings

# Sidebar Controls & Information
with st.sidebar:
    st.markdown("# 🔍")  # Displays a clean emoji icon directly
    st.title("System Status")
    
    
    try:
        df, tfidf_vectorizer, tfidf_matrix, dense_embeddings = load_and_prepare_data()
        st.success("🟢 Dataset & Vector Index Ready")
        st.info(f"📊 **Corpus Size:** {len(df):,} Patent Records")
    except Exception as e:
        st.error(f"🔴 Pipeline Error: {e}")
        st.stop()
        
    st.divider()
    st.markdown("### ⚙️ Pipeline Weights")
    st.write("• **Keyword Match (TF-IDF):** 30%")
    st.write("• **Semantic Search (Transformer):** 70%")
    
    st.divider()
    st.caption("NLP Mini Project System v1.0")

# Main Content Layout
col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("1. Propose Invention Draft")
    user_input = st.text_area(
        "Enter Invention Description / Abstract:",
        height=220,
        placeholder="Describe the technical core, components, and method of your proposed invention here..."
    )
    
    analyze_btn = st.button("🚀 Run Novelty Analysis", type="primary", use_container_width=True)

if analyze_btn:
    if not user_input.strip():
        st.warning("⚠️ Please provide an invention description to perform evaluation.")
    else:
        with st.spinner("Processing NLP Pipeline & Semantic Embedding Distance..."):
            novelty_score, top_matches = compute_novelty_score(
                user_input, df, dense_embeddings, tfidf_vectorizer, tfidf_matrix
            )
            
        with col_results:
            st.subheader("2. Novelty Assessment")
            
            # Metric Card Display
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Calculated Novelty Score</div>
                    <div class="metric-value">{novelty_score}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("") # Spacing
            
            # Status Badge Decision
            if novelty_score >= 70:
                st.success("🟢 **High Novelty Detected**\n\nThe input idea shows minimal overlap with existing prior art in the index.")
            elif novelty_score >= 40:
                st.warning("🟡 **Moderate Overlap Found**\n\nSimilar technological concepts exist in prior art. Claim refining recommended.")
            else:
                st.error("🔴 **Low Novelty / High Overlap**\n\nHigh semantic and keyword similarity detected against registered patents.")

            st.divider()
            st.markdown("### Top Prior Art Matches")
            
            for idx, row in top_matches.iterrows():
                sim_pct = round(row['similarity_score'] * 100, 2)
                
                with st.expander(f"**{row['title']}** — `{sim_pct}% Match`"):
                    st.progress(sim_pct / 100)
                    st.markdown(f"**Patent ID:** `{row['patent_id']}`")
                    st.markdown(f"**Abstract Snippet:**\n> {row['abstract']}")