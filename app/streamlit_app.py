import streamlit as st
import pickle
import os

# -------------------------
# Load Model and Vectorizer
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "log_reg_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "..", "models", "tfidf_vectorizer.pkl")

# Load Logistic Regression model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load TF-IDF vectorizer
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# -------------------------
# Streamlit App UI
# -------------------------
st.title("📰 Fake News Detection")
st.write("Enter news text below to check whether it is REAL or FAKE.")

# User input
user_input = st.text_area("News Text", height=200)

# Predict button
if st.button("Predict"):
    if user_input.strip() != "":
        # Transform input and predict
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        st.markdown(f"### Prediction: **{prediction}**")
    else:
        st.warning("⚠️ Please enter some text to predict.")
