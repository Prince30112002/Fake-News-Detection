import os
import pickle
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_data, preprocess_data

# Base directory set
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save_models():
    # Data load + preprocess
    df = load_data()
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))

    # Passive Aggressive Classifier
    pac = PassiveAggressiveClassifier(max_iter=1000)
    pac.fit(X_train, y_train)
    y_pred_pac = pac.predict(X_test)
    print("Passive Aggressive Classifier Accuracy:", accuracy_score(y_test, y_pred_pac))
    print(classification_report(y_test, y_pred_pac))

    # Save models
    with open(os.path.join(MODEL_DIR, "log_reg_model.pkl"), "wb") as f:
        pickle.dump(log_reg, f)

    with open(os.path.join(MODEL_DIR, "pac_model.pkl"), "wb") as f:
        pickle.dump(pac, f)

    # Save vectorizer
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Models and vectorizer saved successfully in 'models/' folder!")


if __name__ == "__main__":
    train_and_save_models()
