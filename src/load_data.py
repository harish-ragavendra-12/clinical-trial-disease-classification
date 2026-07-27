import pandas as pd

from src.config import PROCESSED_DATA_FILE, RAW_DATA_FILE


def load_dataset() -> pd.DataFrame:
    """
    Load the raw clinical trial dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """

    return pd.read_csv(RAW_DATA_FILE)

def display_dataset_info(df: pd.DataFrame) -> None:
    """
    Display basic information about the dataset.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        None
    """


    print("=" * 60)
    print("Dataset Shape")
    print("=" * 60)
    print(df.shape)


    print("\n" + "=" * 60)
    print("Column Names")
    print("=" * 60)
    print(df.columns.tolist())


    print("\n" + "=" * 60)
    print("Dataset Information")
    print("=" * 60)
    df.info()


    print("\n" + "=" * 60)
    print("Data Types")
    print("=" * 60)
    print(df.dtypes)


    print("\n" + "=" * 60)
    print("Missing Values")
    print("=" * 60)
    print(df.isnull().sum())


    print("\n" + "=" * 60)
    print("Duplicate Records")
    print("=" * 60)
    print(df.duplicated().sum())


    print("\n" + "=" * 60)
    print("Statistical Summary")
    print("=" * 60)
    print(df.describe(include="all"))


    print("\n" + "=" * 60)
    print("Unique Disease Categories")
    print("=" * 60)
    print(df["source_condition_query"].nunique())


    print("\n" + "=" * 60)
    print("Top 10 Disease Categories")
    print("=" * 60)
    print(df["source_condition_query"].value_counts().head(10))

def save_dataset(df: pd.DataFrame) -> None:
    """
    Save the processed dataset to the processed data directory.

    Parameters:
        df (pd.DataFrame): Dataset to save.

    Returns:
        None
    """

    df.to_csv(PROCESSED_DATA_FILE, index=False)

    print(f"\nProcessed dataset saved successfully to:\n{PROCESSED_DATA_FILE}")

def load_data() -> pd.DataFrame:
    """
    Execute the data loading workflow.

    Returns:
        pd.DataFrame: Loaded dataset.
    """

    df = load_dataset()

    display_dataset_info(df)

    save_dataset(df)

    return df

if __name__ == "__main__":
    load_data()
