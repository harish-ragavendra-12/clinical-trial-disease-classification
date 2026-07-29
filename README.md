# 🧬 Clinical Trial Disease Category Classification Using NLP and Machine Learning

An end-to-end **Natural Language Processing (NLP)** and **Machine Learning** project that classifies clinical trial summaries into disease categories. This project preprocesses unstructured medical text, extracts meaningful features using **TF-IDF Vectorization**, trains multiple machine learning models, and deploys the best-performing model through an interactive **Streamlit** web application.

---

## 📌 Project Overview

Clinical trials generate massive amounts of unstructured medical text, including treatment descriptions, eligibility criteria, and disease-related information. Analyzing this information manually is time-consuming and inefficient.

This project automates disease category prediction from the **Brief Summary** of clinical trials using NLP and Machine Learning techniques.

---

## 🎯 Problem Statement

The healthcare industry generates large volumes of clinical trial data in textual format. Since this information is unstructured, extracting meaningful insights and identifying disease categories becomes challenging.

This project aims to:

- Clean and preprocess clinical trial text
- Apply NLP techniques to medical summaries
- Convert text into numerical features using TF-IDF
- Train machine learning models for disease classification
- Build an interactive Streamlit application for disease prediction

---

# ✨ Features

- Medical text preprocessing
- Text cleaning and normalization
- TF-IDF Vectorization
- Disease category prediction
- Multiple machine learning models
- Automatic best model selection
- Interactive Streamlit dashboard
- End-to-end NLP pipeline

---

# 🛠️ Tech Stack

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Plotly
- Joblib
- Streamlit

### Machine Learning

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest Classifier
- Linear Support Vector Classifier (LinearSVC)

### NLP

- Text Cleaning
- Tokenization
- Lemmatization
- Stop Word Removal
- TF-IDF Vectorization

---

# 📂 Project Structure

```
clinical-trial-disease-classification/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── figures/
│
├── models/
│   ├── best_model.pkl
│   ├── vectorizer.pkl
│   └── label_encoder.pkl
│
└── src/
    ├── config.py
    ├── load_data.py
    ├── preprocessing.py
    ├── eda.py
    ├── feature_engineering.py
    ├── model_training.py
    ├── prediction.py
    ├── evaluation.py
    └── utils.py
```

---

# ⚙️ Workflow

```
Clinical Trial Dataset
            │
            ▼
Data Loading
            │
            ▼
Data Preprocessing
            │
            ▼
Exploratory Data Analysis
            │
            ▼
Text Cleaning
            │
            ▼
TF-IDF Vectorization
            │
            ▼
Machine Learning Models
            │
            ▼
Best Model Selection
            │
            ▼
Disease Prediction
            │
            ▼
Streamlit Deployment
```

---

# 🧹 Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Removing duplicate records
- Text cleaning
- Lowercase conversion
- Removing punctuation
- Removing URLs
- Stop-word removal
- Lemmatization
- Feature preparation

---

# 📊 Exploratory Data Analysis

EDA includes:

- Disease category distribution
- Study type analysis
- Study status analysis
- Phase distribution
- Sex distribution
- Summary length analysis
- Word frequency analysis
- Word Cloud generation

---

# 🤖 Machine Learning Models

The following algorithms were trained and compared:

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest Classifier
- Linear Support Vector Classifier

The best-performing model is automatically saved for prediction.

---

# 📈 Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

---

# 🧬 Disease Prediction Pipeline

The prediction workflow:

1. User enters a clinical trial summary.
2. Text is cleaned.
3. TF-IDF Vectorizer converts text into numerical features.
4. Best trained model predicts the disease category.
5. Label Encoder converts prediction into the original disease label.
6. Result is displayed through Streamlit.

---

# 🚀 Streamlit Application

The application allows users to:

- Enter a clinical trial summary
- Predict disease categories
- View prediction results
- Interact with a simple web interface

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/harish-ragavendra-12/clinical-trial-disease-classification.git
```

Move into the project folder

```bash
cd clinical-trial-disease-classification
```

Create virtual environment

```bash
python -m venv .venv
```

Activate virtual environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit Application

```bash
streamlit run app.py
```

---

# 📊 Results

- Successfully cleaned and preprocessed clinical trial text
- Applied TF-IDF feature engineering
- Trained multiple machine learning models
- Built an end-to-end disease classification pipeline
- Developed an interactive Streamlit application
- Automated disease prediction from medical text

---

# 📚 Skills Demonstrated

- Python Programming
- Data Preprocessing
- Natural Language Processing
- Feature Engineering
- TF-IDF Vectorization
- Machine Learning
- Text Classification
- Model Evaluation
- Streamlit Development
- Healthcare Analytics

---

# 🔮 Future Improvements

- Deep Learning models (LSTM, Bi-LSTM)
- Transformer models (BERT, BioBERT)
- Explainable AI (SHAP/LIME)
- Multi-label disease classification
- Clinical Trial Recommendation System
- Cloud deployment

---

# 👨‍💻 Author

**Harish Ragavendra**

Aspiring Data Scientist

GitHub: https://github.com/harish-ragavendra-12

---

# 📄 License

This project is developed for educational and portfolio purposes.