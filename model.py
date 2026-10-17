import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_transformer_model():
    """Cache model loading so it stays in RAM across Streamlit reloads."""
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')

def compute_novelty_score(query_text: str, df: pd.DataFrame, dense_embeddings, tfidf_vectorizer, tfidf_matrix):
    """
    Computes hybrid similarity between proposed invention and stored patent database,
    returning an overall novelty percentage and top matching patents.
    """
    from preprocessing import clean_text
    
    model = load_transformer_model()
    cleaned_query = clean_text(query_text)
    
    # 1. Sparse Similarity (TF-IDF Keyword Matching)
    query_tfidf = tfidf_vectorizer.transform([cleaned_query])
    tfidf_sims = cosine_similarity(query_tfidf, tfidf_matrix)[0]
    
    # 2. Dense Semantic Similarity (Sentence-Transformers)
    query_dense = model.encode([cleaned_query])
    dense_sims = cosine_similarity(query_dense, dense_embeddings)[0]
    
    # Hybrid Similarity Score (30% Keywords + 70% Contextual Meaning)
    hybrid_sims = 0.3 * tfidf_sims + 0.7 * dense_sims
    
    # Copy dataset and attach scores
    results_df = df.copy()
    results_df['similarity_score'] = hybrid_sims
    
    # Sort to retrieve top 5 matches
    top_matches = results_df.sort_values(by='similarity_score', ascending=False).head(5)
    
    # Calculate Novelty Percentage: 100% - Max Similarity %
    max_sim = top_matches['similarity_score'].iloc[0] if not top_matches.empty else 0.0
    novelty_score = round(max(0.0, (1.0 - max_sim) * 100), 2)
    
    return novelty_score, top_matches