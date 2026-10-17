import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pypdf import PdfReader

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def extract_text_from_file(uploaded_file) -> str:
    """Extracts raw text from uploaded TXT or PDF files."""
    if uploaded_file is None:
        return ""
    
    file_type = uploaded_file.name.split('.')[-1].lower()
    extracted_text = ""
    
    if file_type == 'txt':
        extracted_text = uploaded_file.read().decode('utf-8', errors='ignore')
    elif file_type == 'pdf':
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + " "
                
    return extracted_text

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