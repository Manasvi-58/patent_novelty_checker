import streamlit as st
import pandas as pd
import numpy as np
from preprocessing import clean_text, build_tfidf_matrix, extract_text_from_file
from model import compute_novelty_score

# Page Configuration
st.set_page_config(
    page_title="Patent Novelty Checker",
    page_icon="🔍",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .title-text {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
    }
    .subtitle-text {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
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

# Sidebar Controls
with st.sidebar:
    st.markdown("# 🔍")
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
    st.caption("NLP Mini Project System")

# Main Interface Layout
col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("1. Provide Invention Details")
    
    # Input Option Tabs: Paste Text vs Upload File
    tab_text, tab_file = st.tabs(["✍️ Paste Description", "📁 Upload File (.txt / .pdf)"])
    
    user_input = ""
    
    with tab_text:
        text_query = st.text_area(
            "Enter Invention Description / Abstract:",
            height=200,
            placeholder="Describe the technical core, components, and method of your proposed invention here..."
        )
        if text_query.strip():
            user_input = text_query
            
    with tab_file:
        uploaded_file = st.file_uploader(
            "Upload Patent Document (.txt or .pdf):",
            type=["txt", "pdf"]
        )
        if uploaded_file is not None:
            extracted_text = extract_text_from_file(uploaded_file)
            if extracted_text.strip():
                st.success(f"✅ Extracted {len(extracted_text.split())} words from **{uploaded_file.name}**")
                with st.expander("Preview Extracted Document Text"):
                    st.write(extracted_text[:600] + "...")
                user_input = extracted_text
            else:
                st.error("Could not read text from the uploaded file.")

    analyze_btn = st.button("🚀 Run Novelty Analysis", type="primary", use_container_width=True)

if analyze_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter a description or upload a file before analyzing.")
    else:
        with st.spinner("Processing NLP Pipeline & Computing Embeddings Distance..."):
            novelty_score, top_matches = compute_novelty_score(
                user_input, df, dense_embeddings, tfidf_vectorizer, tfidf_matrix
            )
            
        with col_results:
            st.subheader("2. Novelty Assessment")
            
            # Metric Card Display
            st.metric(label="Calculated Novelty Score", value=f"{novelty_score}%")
            
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