import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder

from src.config import (
    CLEANED_DATA_FILE,
    FEATURE_ENGINEERED_DATA_FILE,
    LABEL_ENCODER_FILE
)


# ==========================================================
# Load Cleaned Dataset
# ==========================================================

def load_cleaned_dataset():
    """
    Load the cleaned clinical trial dataset.

    Returns:
        pd.DataFrame
    """

    print("=" * 60)
    print("Loading Cleaned Dataset")
    print("=" * 60)

    df = pd.read_csv(CLEANED_DATA_FILE)

    print(f"Dataset Shape : {df.shape}")

    return df


# ==========================================================
# Select Feature and Target
# ==========================================================

def select_features_and_target(df):
    """
    Select feature and target columns.

    Parameters:
        df (pd.DataFrame)

    Returns:
        tuple:
            X
            y
    """

    print("=" * 60)
    print("Selecting Features and Target")
    print("=" * 60)

    X = df["brief_summary"]

    y = df["source_condition_query"]

    print("Feature Column : brief_summary")
    print("Target Column  : source_condition_query")

    return X, y


# ==========================================================
# Encode Target Labels
# ==========================================================

def encode_target_labels(y):
    """
    Encode disease categories.

    Parameters:
        y (pd.Series)

    Returns:
        tuple:
            encoded_labels
            label_encoder
    """

    print("=" * 60)
    print("Encoding Target Labels")
    print("=" * 60)

    label_encoder = LabelEncoder()

    encoded_labels = label_encoder.fit_transform(y)

    print(f"Total Classes : {len(label_encoder.classes_)}")

    return encoded_labels, label_encoder


# ==========================================================
# Save Label Encoder
# ==========================================================

def save_label_encoder(label_encoder):
    """
    Save LabelEncoder.
    """

    print("=" * 60)
    print("Saving Label Encoder")
    print("=" * 60)

    joblib.dump(
        label_encoder,
        LABEL_ENCODER_FILE
    )

    print("Label Encoder saved successfully.")


# ==========================================================
# Create Feature Engineered Dataset
# ==========================================================

def create_feature_engineered_dataset(
        X,
        encoded_labels
):
    """
    Create feature engineered dataframe.

    Parameters:
        X
        encoded_labels

    Returns:
        pd.DataFrame
    """

    print("=" * 60)
    print("Creating Feature Engineered Dataset")
    print("=" * 60)

    feature_engineered_df = pd.DataFrame({

        "cleaned_summary": X,

        "encoded_conditions": encoded_labels

    })

    print(feature_engineered_df.head())

    return feature_engineered_df


# ==========================================================
# Save Feature Engineered Dataset
# ==========================================================

def save_feature_engineered_dataset(df):
    """
    Save feature engineered dataset.
    """

    print("=" * 60)
    print("Saving Feature Engineered Dataset")
    print("=" * 60)

    df.to_csv(
        FEATURE_ENGINEERED_DATA_FILE,
        index=False
    )

    print(
        f"Dataset saved successfully:\n{FEATURE_ENGINEERED_DATA_FILE}"
    )


# ==========================================================
# Feature Engineering Pipeline
# ==========================================================

def feature_engineering():
    """
    Execute feature engineering pipeline.

    Returns:
        pd.DataFrame
    """

    print("=" * 60)
    print("Starting Feature Engineering Pipeline")
    print("=" * 60)

    # Load cleaned dataset
    df = load_cleaned_dataset()

    # Select feature and target
    X, y = select_features_and_target(df)

    # Encode target labels
    encoded_labels, label_encoder = encode_target_labels(y)

    # Save encoder
    save_label_encoder(label_encoder)

    # Create feature engineered dataframe
    feature_engineered_df = create_feature_engineered_dataset(
        X,
        encoded_labels
    )

    # Save dataset
    save_feature_engineered_dataset(
        feature_engineered_df
    )

    print("=" * 60)
    print("Feature Engineering Pipeline Completed Successfully")
    print("=" * 60)

    return feature_engineered_df


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    feature_engineering()
