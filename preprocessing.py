import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Preprocesses raw text by lowercasing, removing noise, removing stopwords, and lemmatizing."""
    if not isinstance(text, str):
        return ""
    
    # Remove non-alphabetical characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenize and filter stop words + lemmatize
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(cleaned_tokens)

def build_tfidf_matrix(df: pd.DataFrame, text_column: str = 'cleaned_abstract'):
    """Generates a TF-IDF Document-Term Matrix for the corpus."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[text_column])
    return vectorizer, tfidf_matrix