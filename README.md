# Patent Novelty Checker

A Streamlit app that compares a proposed invention abstract with patent records using TF-IDF and Sentence-Transformer similarity.

## Setup

From the project directory in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app uses `dataset/patent_dataset.csv` and `dataset/embeddings.npy`. To regenerate them, place the source file at `dataset/train_mini.csv` and run:

```powershell
python prepare_data.py
```

The raw `train_mini.csv` is excluded from Git because it exceeds GitHub's regular 100 MB file limit. It remains local and is not removed by this project setup.
# 🔍 Patent Novelty Checker

A Natural Language Processing (NLP) application designed to evaluate the novelty of proposed invention descriptions against existing patent abstracts using a hybrid approach (30% TF-IDF keyword matching + 70% Sentence-Transformer semantic search).

## 🚀 How to Run the Project Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/patent_novelty_checker.git](https://github.com/YOUR_USERNAME/patent_novelty_checker.git)
cd patent_novelty_checker# patent_novelty_checker
