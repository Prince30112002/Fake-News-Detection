import pandas as pd
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
RAW_PATH = r"C:\Users\P.R\FakeNewsDetection\data\raw\train.csv"
PROCESSED_PATH = r"C:\Users\P.R\FakeNewsDetection\data\processed\processed_data.csv"
VECTORIZER_PATH = r"C:\Users\P.R\FakeNewsDetection\models\tfidf_vectorizer.pkl"

# Load dataset
df = pd.read_csv(RAW_PATH)
print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# Text cleaning function
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)  # keep only letters
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    return text.lower().strip()

# Apply cleaning
df["clean_text"] = df["text"].apply(clean_text)

# Save processed data
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)
print("Preprocessing done ✅")

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
vectorizer.fit(df["clean_text"])

# Save vectorizer
os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print("Vectorizer saved ✅")
