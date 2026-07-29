import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from src.config import (
    FEATURE_ENGINEERED_DATA_FILE,
    LABEL_ENCODER_FILE,
    BEST_MODEL_FILE,
    VECTORIZER_FILE
)

from src.model_selection import compare_models


# ==========================================================
# Load Feature Engineered Dataset
# ==========================================================

def load_feature_engineered_dataset():
    """
    Load feature engineered dataset.

    Returns:
        pd.DataFrame
    """

    print("=" * 60)
    print("Loading Feature Engineered Dataset")
    print("=" * 60)

    df = pd.read_csv(
        FEATURE_ENGINEERED_DATA_FILE
    )

    print(f"Dataset Shape : {df.shape}")

    return df


# ==========================================================
# Load Label Encoder
# ==========================================================

def load_label_encoder():
    """
    Load saved LabelEncoder.

    Returns:
        LabelEncoder
    """

    return joblib.load(
        LABEL_ENCODER_FILE
    )


# ==========================================================
# Split Dataset
# ==========================================================

def split_dataset(df):
    """
    Split dataset into train and test sets.
    """

    X = df["cleaned_summary"]

    y = df["encoded_conditions"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ==========================================================
# Vectorize Text
# ==========================================================

def vectorize_text(
        X_train,
        X_test
):
    """
    TF-IDF Vectorization.
    """

    vectorizer = TfidfVectorizer()

    X_train_vectorized = vectorizer.fit_transform(
        X_train
    )

    X_test_vectorized = vectorizer.transform(
        X_test
    )

    return (
        X_train_vectorized,
        X_test_vectorized,
        vectorizer
    )


# ==========================================================
# Logistic Regression
# ==========================================================

def train_logistic_regression(
        X_train_vectorized,
        y_train
):

    model = LogisticRegression()

    model.fit(
        X_train_vectorized,
        y_train
    )

    return model


# ==========================================================
# Multinomial Naive Bayes
# ==========================================================

def train_multinomial_naive_bayes(
        X_train_vectorized,
        y_train
):

    model = MultinomialNB()

    model.fit(
        X_train_vectorized,
        y_train
    )

    return model


# ==========================================================
# Random Forest
# ==========================================================

def train_random_forest(
        X_train_vectorized,
        y_train
):

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(
        X_train_vectorized,
        y_train
    )

    return model


# ==========================================================
# Linear SVC
# ==========================================================

def train_linear_svc(
        X_train_vectorized,
        y_train
):

    model = LinearSVC()

    model.fit(
        X_train_vectorized,
        y_train
    )

    return model


# ==========================================================
# Save Artifacts
# ==========================================================

def save_artifacts(
        best_model,
        vectorizer
):
    """
    Save trained artifacts.
    """

    joblib.dump(
        best_model,
        BEST_MODEL_FILE
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE
    )

    print("\nArtifacts saved successfully.")


# ==========================================================
# Training Pipeline
# ==========================================================

def training_pipeline():

    print("=" * 60)
    print("Starting Model Training Pipeline")
    print("=" * 60)

    # Load feature engineered dataset
    df = load_feature_engineered_dataset()

    # Load label encoder
    label_encoder = load_label_encoder()

    # Split dataset
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_dataset(df)

    # Vectorize text
    (
        X_train_vectorized,
        X_test_vectorized,
        vectorizer
    ) = vectorize_text(
        X_train,
        X_test
    )

    # ==============================
    # Train Models
    # ==============================

    logistic_model = train_logistic_regression(
        X_train_vectorized,
        y_train
    )

    mnb_model = train_multinomial_naive_bayes(
        X_train_vectorized,
        y_train
    )

    random_forest_model = train_random_forest(
        X_train_vectorized,
        y_train
    )

    linear_svc_model = train_linear_svc(
        X_train_vectorized,
        y_train
    )

    # ==============================
    # Compare Models
    # ==============================

    best_model = compare_models(
        logistic_model,
        mnb_model,
        random_forest_model,
        linear_svc_model,
        X_test_vectorized,
        y_test
    )

    # Save artifacts
    save_artifacts(
        best_model,
        vectorizer
    )

    print("=" * 60)
    print("Model Training Pipeline Completed Successfully")
    print("=" * 60)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    training_pipeline()
