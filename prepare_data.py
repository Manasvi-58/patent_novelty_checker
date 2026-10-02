import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from preprocessing import clean_text

print("1. Reading dataset (300 rows for instant performance)...")
# Load 300 rows only - ultra-fast execution on any laptop
df = pd.read_csv("dataset/train_mini.csv", nrows=300)

if 'publication_number' in df.columns:
    df['patent_id'] = df['publication_number']

if 'title' not in df.columns:
    df['title'] = "Patent Record " + df['patent_id'].astype(str)

processed_df = df[['patent_id', 'title', 'abstract']].dropna(subset=['abstract'])

print("2. Preprocessing text...")
processed_df['cleaned_abstract'] = processed_df['abstract'].apply(clean_text)

# Save processed dataset
processed_df.to_csv("dataset/patent_dataset.csv", index=False)

print("3. Pre-computing embeddings (takes < 5 seconds)...")
model = SentenceTransformer('paraphrase-MiniLM-L3-v2') # 3x faster & lightweight model
embeddings = model.encode(processed_df['cleaned_abstract'].tolist(), show_progress_bar=True)

# Save pre-computed embeddings
np.save("dataset/embeddings.npy", embeddings)
print("Done! Files saved successfully.")