import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import classification_report, accuracy_score

# Paths
PROCESSED_PATH = r"C:\Users\P.R\FakeNewsDetection\data\processed\processed_data.csv"
VECTORIZER_PATH = r"C:\Users\P.R\FakeNewsDetection\models\tfidf_vectorizer.pkl"
MODELS_DIR = r"C:\Users\P.R\FakeNewsDetection\models"

print("Loading processed data...")
df = pd.read_csv(PROCESSED_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# Fix NaN values
df["clean_text"] = df["clean_text"].fillna("")

X = df["clean_text"]
y = df["label"]

# Load TF-IDF vectorizer
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

X_vec = vectorizer.transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Passive Aggressive Classifier
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
pac.fit(X_train, y_train)
y_pred_pac = pac.predict(X_test)
print("\nPassive Aggressive Classifier Accuracy:", accuracy_score(y_test, y_pred_pac))
print(classification_report(y_test, y_pred_pac))

# Save models
os.makedirs(MODELS_DIR, exist_ok=True)

with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "wb") as f:
    pickle.dump(log_reg, f)

with open(os.path.join(MODELS_DIR, "passive_aggressive.pkl"), "wb") as f:
    pickle.dump(pac, f)

print("\n✅ Models and vectorizer saved successfully in 'models/' folder!")
