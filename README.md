# Sentiment Analysis Through Text

This project is a simple **Sentiment Analysis** tool built using Python, `scikit-learn`, and `NLTK`.  
It preprocesses text, trains a Support Vector Machine (SVM) model, and predicts whether a given sentence is positive, negative, or neutral.

---

## 📂 Project Structure

sentiment_AnalysisThroghText/
│
├── sentiment.py # Main script to train and test the model
├── sample_sentiment_dataset.csv # Dataset for training (text + label)
├── sentiment_model.pkl # Saved trained model
├── vectorizer.pkl # Saved TF-IDF vectorizer
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 📌 Features
- **Text Preprocessing:**
  - Lowercasing
  - Tokenization
  - Stopword removal
  - Lemmatization
- **Feature Extraction:** TF-IDF vectorization  
- **Model:** SVM (Support Vector Machine) with linear kernel  
- Saves trained model and vectorizer for future predictions  
- Example prediction after training  

---

## 📦 Installation

1. **Clone this repository**
```bash
git clone https://github.com/your-username/sentiment_AnalysisThroghText.git
cd sentiment_AnalysisThroghText

2. Install dependencies
pip install -r requirements.txt

3. pip install -r requirements.txt
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

