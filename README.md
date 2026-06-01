# 🎭 Sentiment Analysis on Movie Reviews

A complete NLP pipeline comparing multiple machine learning models for binary sentiment classification using the NLTK Movie Reviews dataset.

---

## 📌 Overview

This project builds and evaluates a text classification system that determines whether a movie review is **positive** or **negative**. It covers the full ML workflow — from raw text to a working prediction function — and is designed to be readable and educational.

---

## 🧪 Models Compared

| Model | Feature Set |
|---|---|
| Naive Bayes | BoW / TF-IDF |
| Logistic Regression | BoW / TF-IDF |
| Linear SVM | BoW / TF-IDF |
| Random Forest | BoW / TF-IDF |

---

## 🔧 Pipeline

```
Raw Text
   ↓
Preprocessing (lowercase → clean → tokenize → remove stopwords → lemmatize)
   ↓
Feature Extraction (Bag of Words  |  TF-IDF with bigrams)
   ↓
Model Training (4 classifiers × 2 feature sets)
   ↓
Evaluation (accuracy, precision, recall, F1, confusion matrix)
   ↓
Error Analysis + Live Prediction
```

---

## 📊 Results

> *Run the notebook to generate results. Logistic Regression + TF-IDF typically achieves ~85–87% accuracy.*

Output charts saved to `/outputs/`:
- `data_exploration.png` — class distribution & review length
- `model_comparison.png` — accuracy comparison across all models
- `best_model_analysis.png` — confusion matrix + most predictive words

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/mdarifhasanbadsha/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook sentiment_analysis.ipynb
```

---

## 📦 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
jupyter
```

---

## 📁 Project Structure

```
sentiment-analysis/
│
├── sentiment_analysis.ipynb   # Main notebook
├── requirements.txt
├── README.md
│
├── outputs/                   # Generated plots
│   ├── data_exploration.png
│   ├── model_comparison.png
│   └── best_model_analysis.png
│
└── models/                    # (optional) saved model files
```

---

## 💡 Key Learnings

- **TF-IDF outperforms Bag of Words** by weighting rare but meaningful terms
- **Logistic Regression** is a strong baseline for text classification
- **Bigrams** (e.g., "not good") capture negation better than unigrams alone
- Error analysis reveals the model struggles most with **sarcasm** and **mixed-opinion** reviews

---

## 🔭 Future Work

- [ ] Fine-tune DistilBERT for improved contextual understanding
- [ ] Add cross-validation scores to results table
- [ ] Deploy as a simple web app (Streamlit or Flask)
- [ ] Expand to multi-class sentiment (5-star ratings)

---

## 👤 Author

**Md Arif Hasan Badsha**  
Final-year B.Sc. AI Student  
[GitHub](https://github.com/mdarifhasanbadsha)

---

## 📜 Dataset

Pang, B. and Lee, L. (2002). *A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts.* Proceedings of ACL.  
Available via `nltk.corpus.movie_reviews`
