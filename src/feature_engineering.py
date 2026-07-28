import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder

from src.config import (
    CLEANED_DATA_FILE,
    LABEL_ENCODER_FILE
)


def load_cleaned_dataset() -> pd.DataFrame:
    """
    Load the cleaned clinical trial dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """

    print("=" * 60)
    print("Loading Cleaned Dataset")
    print("=" * 60)

    df = pd.read_csv(CLEANED_DATA_FILE)

    print(f"Dataset Shape: {df.shape}")

    return df


def select_features_and_target(df: pd.DataFrame):
    """
    Select input feature and target variable.

    Feature:
        brief_summary

    Target:
        source_condition_query

    Parameters:
        df (pd.DataFrame): Cleaned clinical trial dataset.

    Returns:
        tuple:
            X: Text feature.
            y: Disease category labels.
    """

    print("=" * 60)
    print("Selecting Features and Target")
    print("=" * 60)

    X = df["brief_summary"]

    y = df["source_condition_query"]

    print(f"Feature Column : brief_summary")
    print(f"Target Column  : source_condition_query")

    return X, y


def encode_target_labels(y: pd.Series):
    """
    Encode disease categories into numerical labels.

    Parameters:
        y (pd.Series): Target labels.

    Returns:
        tuple:
            encoded_y: Encoded labels.
            encoder: Fitted LabelEncoder.
    """

    print("=" * 60)
    print("Encoding Target Labels")
    print("=" * 60)

    encoder = LabelEncoder()

    encoded_y = encoder.fit_transform(y)

    print(f"Number of Classes: {len(encoder.classes_)}")

    return encoded_y, encoder


def save_label_encoder(encoder):
    """
    Save LabelEncoder artifact.

    Parameters:
        encoder: Fitted LabelEncoder.

    Returns:
        None
    """

    print("=" * 60)
    print("Saving Label Encoder")
    print("=" * 60)

    joblib.dump(
        encoder,
        LABEL_ENCODER_FILE
    )

    print(
        f"Label Encoder saved successfully:\n{LABEL_ENCODER_FILE}"
    )


def feature_engineering():
    """
    Execute feature engineering pipeline.

    Returns:
        tuple:
            X: Text feature.
            encoded_y: Encoded target labels.
    """

    print("=" * 60)
    print("Starting Feature Engineering Pipeline")
    print("=" * 60)

    # Load dataset
    df = load_cleaned_dataset()

    # Select features and target
    X, y = select_features_and_target(df)

    # Encode target labels
    encoded_y, encoder = encode_target_labels(y)

    # Save encoder
    save_label_encoder(encoder)

    print("=" * 60)
    print("Feature Engineering Completed")
    print("=" * 60)

    return X, encoded_y

if __name__ == "__main__":
    feature_engineering()
