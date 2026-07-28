import pandas as pd

import joblib

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CLEANED_DATA_FILE, VECTORIZER_FILE, LABEL_ENCODER_FILE


def load_cleaned_dataset() -> pd.DataFrame:
    """
    Load the cleaned clinical trial dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """

    return pd.read_csv(CLEANED_DATA_FILE)

def select_features_and_target(df: pd.DataFrame):
    """
    Select the input feature and target variable for the
    disease classification model.

    Parameters:
        df (pd.DataFrame): Cleaned clinical trial dataset.

    Returns:
        tuple:
            X (pd.Series): Clinical trial summaries.
            y (pd.Series): Disease categories.
    """

    X = df["brief_summary"]
    y = df["source_condition_query"]

    return X, y

def encode_target_labels(y: pd.Series):
    """
    Encode the target disease categories into numerical labels.

    Parameters:
        y (pd.Series): Target disease categories.

    Returns:
        tuple:
            encoded_y (numpy.ndarray): Encoded target labels.
            encoder (LabelEncoder): Fitted label encoder.
    """

    encoder = LabelEncoder()

    encoded_y = encoder.fit_transform(y)

    return encoded_y, encoder

def split_dataset(X: pd.Series, encoded_y):
    """
    Split the feature and target datasets into training and testing sets.

    Parameters:
        X (pd.Series): Clinical trial summaries.
        encoded_y (numpy.ndarray): Encoded target labels.

    Returns:
        tuple:
            X_train (pd.Series): Training feature set.
            X_test (pd.Series): Testing feature set.
            y_train (numpy.ndarray): Training target labels.
            y_test (numpy.ndarray): Testing target labels.
    """

    print("=" * 60)
    print("Splitting Dataset")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        encoded_y,
        test_size=0.20,
        random_state=42,
        stratify=encoded_y
    )

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    return X_train, X_test, y_train, y_test

def vectorize_text(X_train: pd.Series, X_test: pd.Series):
    """
    Convert the training and testing text data into TF-IDF feature vectors.

    Parameters:
        X_train (pd.Series): Training clinical trial summaries.
        X_test (pd.Series): Testing clinical trial summaries.

    Returns:
        tuple:
            X_train_vectorized (scipy.sparse.csr_matrix): Vectorized training features.
            X_test_vectorized (scipy.sparse.csr_matrix): Vectorized testing features.
            vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
    """

    print("=" * 60)
    print("Vectorizing Text using TF-IDF")
    print("=" * 60)

    vectorizer = TfidfVectorizer()

    X_train_vectorized = vectorizer.fit_transform(X_train)

    X_test_vectorized = vectorizer.transform(X_test)

    print(f"Training Feature Matrix Shape: {X_train_vectorized.shape}")
    print(f"Testing Feature Matrix Shape : {X_test_vectorized.shape}")

    return  X_train_vectorized, X_test_vectorized, vectorizer

def save_artifacts(vectorizer, encoder) -> None:
    """
    Save the fitted TF-IDF vectorizer and LabelEncoder for future use.

    Parameters:
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        encoder (LabelEncoder): Fitted label encoder.

    Returns:
        None
    """

    print("=" * 60)
    print("Saving Feature Engineering Artifacts")
    print("=" * 60)

    # Save TF-IDF Vectorizer
    joblib.dump(vectorizer, VECTORIZER_FILE)

    print(f"TF-IDF Vectorizer saved successfully to:\n{VECTORIZER_FILE}")

    # Save Label Encoder
    joblib.dump(encoder, LABEL_ENCODER_FILE)

    print(f"Label Encoder saved successfully to:\n{LABEL_ENCODER_FILE}")

def feature_engineering():

    df = load_cleaned_dataset()
    
    X, y = select_features_and_target(df)
    
    encoded_y, encoder = encode_target_labels(y)

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        encoded_y
    )

    # Vectorize text
    X_train_vectorized, X_test_vectorized, vectorizer = vectorize_text(
        X_train,
        X_test
    )

    # Save artifacts
    save_artifacts(vectorizer, encoder)

    return(
        X_train_vectorized,
        X_test_vectorized,
        y_train,
        y_test,
        vectorizer,
        encoder
    )

if __name__ == "__main__":
    feature_engineering()
