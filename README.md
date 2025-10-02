
---

# 📰 Fake News Detection

Detecting fake news articles using Machine Learning models with Python and Streamlit.

---

## 📌 Table of Contents

* <a href="#overview">Overview</a>
* <a href="#business-problem">Business Problem</a>
* <a href="#dataset">Dataset</a>
* <a href="#tools--technologies">Tools & Technologies</a>
* <a href="#project-structure">Project Structure</a>
* <a href="#data-cleaning--preprocessing">Data Cleaning & Preprocessing</a>
* <a href="#model-training">Model Training</a>
* <a href="#deployment">Deployment</a>
* <a href="#how-to-run-this-project">How to Run This Project</a>
* <a href="#final-recommendations">Final Recommendations</a>
* <a href="#author--contact">Author & Contact</a>

---

<h2><a class="anchor" id="overview"></a>Overview</h2>

This project classifies news articles into **FAKE** or **REAL** categories. It uses a combination of **text preprocessing, TF-IDF vectorization**, and **machine learning models** (Logistic Regression, Passive Aggressive Classifier). The model is deployed using **Streamlit** for interactive predictions.

---

<h2><a class="anchor" id="business-problem"></a>Business Problem</h2>

Fake news spreads rapidly online, affecting public opinion and decision-making. This project aims to:

* Automatically classify news as fake or real
* Improve detection speed compared to manual verification
* Provide a simple interface for journalists or researchers
* Analyze key features contributing to classification

---

<h2><a class="anchor" id="dataset"></a>Dataset</h2>

* `train.csv` file containing columns: `title`, `text`, `label`
* Label values: **FAKE** or **REAL**
* Located in `/data/raw/` folder

---

<h2><a class="anchor" id="tools--technologies"></a>Tools & Technologies</h2>

* Python (Pandas, scikit-learn, Matplotlib, Seaborn)
* Streamlit (Web Deployment)
* GitHub (Version Control)

---

<h2><a class="anchor" id="project-structure"></a>Project Structure</h2>

```
fake-news-detection/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── train.csv
│   └── processed/
│
├── src/
│   ├── data_preprocessing.py
│   └── model_training.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── passive_aggressive.pkl
│   └── tfidf_vectorizer.pkl
│
└── app/
    └── streamlit_app.py
```

---

<h2><a class="anchor" id="data-cleaning--preprocessing"></a>Data Cleaning & Preprocessing</h2>

* Removed null/empty values from `title` and `text`
* Converted text to lowercase, removed special characters, and extra spaces
* Created a `clean_text` column for modeling
* Saved processed data and TF-IDF vectorizer in `/models/`

---

<h2><a class="anchor" id="model-training"></a>Model Training</h2>

* **Features:** TF-IDF vectorized `clean_text`
* **Models trained:**

  * Logistic Regression (Accuracy ~91%)
  * Passive Aggressive Classifier (Accuracy ~92%)
* **Train/Test split:** 80/20
* Models saved as `.pkl` files for deployment

---

<h2><a class="anchor" id="deployment"></a>Deployment</h2>

* Streamlit web app allows users to input news text
* Predicts **FAKE** or **REAL** with probability scores
* Interactive and user-friendly interface

---

<h2><a class="anchor" id="how-to-run-this-project"></a>How to Run This Project</h2>

1. Clone the repository:

```bash
git clone https://github.com/yourusername/fake-news-detection.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run Streamlit app:

```bash
"C:/Program Files/Python313/python.exe" -m streamlit run app/streamlit_app.py
```

4. Input news text in the browser to predict **FAKE** or **REAL**

---

<h2><a class="anchor" id="final-recommendations"></a>Final Recommendations</h2>

* Integrate model with news websites or social media platforms for real-time detection
* Continuously retrain model with new articles to improve accuracy
* Extend features using NLP techniques like Word2Vec, BERT for better classification

---

<h2><a class="anchor" id="author--contact"></a>Author & Contact</h2>

*Prince Rajak*
Computer Science Student / ML Enthusiast
📧 Email: [rajakprince30112002@gmail.com](mailto:rajakprince30112002@gmail.com)


---


