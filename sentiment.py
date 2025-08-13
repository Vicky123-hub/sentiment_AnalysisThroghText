import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import re
import nltk
import pickle
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.stem import WordNetLemmatizer


df=pd.read_csv(r'sample_sentiment_dataset.csv')

texts=df.iloc[:,0].tolist()

labels=df.iloc[:,1].tolist()

def preprocessed(text):
  text=text.lower()
  tokens=word_tokenize(text)
  sw=set(stopwords.words('english'))
  text=[word for word in tokens if word not in sw and word not in punctuation]
  wnl=WordNetLemmatizer()
  text=[wnl.lemmatize(word) for word in text]
  return " ".join(text)

cleaned_texts=[preprocessed(word) for word in texts]
tfidf=TfidfVectorizer()
X=tfidf.fit_transform(cleaned_texts).toarray()
Y=np.array(labels)

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

model=SVC(kernel='linear')
model.fit(X_train,Y_train)
print(model.score(X_test, Y_test))

with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Model and vectorizer saved successfully!")

sample_review = ["vicky is good boy"]
sample_clean = preprocessed(sample_review[0])
sample_vectorArray = tfidf.transform([sample_clean]).toarray()

print("Prediction:", model.predict(sample_vectorArray)[0])