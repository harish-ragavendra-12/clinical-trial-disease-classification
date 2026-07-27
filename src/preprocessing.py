import pandas as pd
import re

import regex

from src.config import PROCESSED_DATA_FILE, CLEANED_DATA_FILE


def load_processed_dataset() -> pd.DataFrame:
    """
    Load the processed clinical trial dataset.

    Returns:
        pd.DataFrame: Processed dataset.
    """

    return pd.read_csv(PROCESSED_DATA_FILE)

def handle_missing_values(df: pd.DataFrame) -> None:
    """
    Handle missing values in the dataset.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset after handling missing values.
    """

    print("=" * 60)
    print("Handling Missing Values")
    print("=" * 60)

    print("\nMissing values before handling:")
    print(df.isnull().sum())

    rows_before = len(df)

    # Remove rows with missing text or target labels
    df = df.dropna(subset=["brief_summary","source_condition_query"])

    rows_after = len(df)

    print("\nMissing values after handling:")
    print(df.isnull().sum())

    print(f"\nRows removed: {rows_before - rows_after}")

    return df

def remove_duplicates(df: pd.DataFrame) -> None:
    """
    Remove duplicate rows from the dataset.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset after removing duplicates.
    """

    print("=" * 60)
    print("Removing Duplicate Records")
    print("=" * 60)

    duplicate_count = df.duplicated().sum()
    print(f"Duplicate records before removal: {duplicate_count}")

    df = df.drop_duplicates()

    duplicate_count = df.duplicated().sum()
    print(f"Duplicate records after removal: {duplicate_count}")

    return df

def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the clinical trial text.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with cleaned text.
    """

    print("=" * 60)
    print("Cleaning Text")
    print("=" * 60)

    df["brief_summary"] = (
        df["brief_summary"]
        .str.lower()
        .str.replace(r"<.*?>", "", regex=True)
        .str.replace(r"http\S+|www\S+", "", regex=True)
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    print("Text cleaning completed.")

    return df

def save_cleaned_dataset(df: pd.DataFrame) -> None:
    """
    Save the cleaned dataset to the processed data directory.

    Parameters:
        df (pd.DataFrame): Dataset to save.

    Returns:
        None
    """

    df.to_csv(CLEANED_DATA_FILE, index=False)

    print(f"\nCleaned dataset saved successfully to:\n{CLEANED_DATA_FILE}")

def preprocessing() -> pd.DataFrame:
    """
    Execute the complete data preprocessing workflow.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """

    df = load_processed_dataset()

    df = handle_missing_values(df)

    df = remove_duplicates(df)

    df = clean_text(df)

    save_cleaned_dataset(df)

    return df

if __name__ == "__main__":
    preprocessing()
