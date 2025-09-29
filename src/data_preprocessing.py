# src/data_preprocessing.py

import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------
# File paths
DATA_PATH = "data/raw/train.csv"
PROCESSED_PATH = "data/processed/"
# -------------------------

# Load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# -------------------------
# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.strip()
    return text

df['text'] = df['text'].apply(preprocess_text)
df['title'] = df['title'].apply(preprocess_text)

print("Preprocessing done ✅")

# -------------------------
# Train-test split
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save vectorizer
with open(PROCESSED_PATH + "tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Vectorizer saved ✅")
