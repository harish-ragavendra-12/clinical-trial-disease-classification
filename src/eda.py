import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter
from wordcloud import WordCloud

from src.config import CLEANED_DATA_FILE, FIGURES_DIR


def load_cleaned_dataset() -> pd.DataFrame:
    """
    Load the cleaned clinical trial dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """

    return pd.read_csv(CLEANED_DATA_FILE)

def dataset_summary(df: pd.DataFrame) -> None:
    """
    Display a summary of the cleaned dataset.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        None
    """

    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    print(f"\nDataset Shape: {df.shape}")

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

def plot_disease_distribution(df: pd.DataFrame) -> None:
    """
    Plot the top 20 disease categories.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        None
    """

    print("=" * 60)
    print("Disease Category Distribution")
    print("=" * 60)

    disease_counts = (df["source_condition_query"].value_counts().head(20))

    plt.figure(figsize=(12, 6))

    disease_counts.plot(kind="bar")

    plt.title("Top 20 Disease Categories")

    plt.xlabel("Disease Category")
    plt.ylabel("Number of Clinical Trials")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "disease_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Disease distribution plot saved successfully.")

def plot_study_status_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of study status.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        None
    """

    print("=" * 60)
    print("Study Status Distribution")
    print("=" * 60)

    study_status_counts = (df["overall_status"].value_counts())

    plt.figure(figsize=(12, 6))

    study_status_counts.plot(kind="bar")

    plt.title("Study Status Distribution")

    plt.xlabel("Study Status")
    plt.ylabel("Number of Clinical Trials")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "study_status_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Study Status distribution plot saved successfully.")

def plot_study_type_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of study types.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        None
    """

    print("=" * 60)
    print("Study Type Distribution")
    print("=" * 60)

    study_type_counts = (df["study_type"].value_counts())

    plt.figure(figsize=(12, 6))

    study_type_counts.plot(kind="bar")

    plt.title("Study Type Distribution")

    plt.xlabel("Study Type")
    plt.ylabel("Number of Clinical Trials")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "study_type_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Study type distribution plot saved successfully.")

def plot_phase_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of clinical trial phases.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        None
    """

    print("=" * 60)
    print("Clinical Trial Phase Distribution")
    print("=" * 60)

    phase_counts = (df["phase"].fillna("Not Specified").value_counts())

    plt.figure(figsize=(12, 6))

    phase_counts.plot(kind="bar")

    plt.title("Clinical Trial Phase Distribution")

    plt.xlabel("Clinical Trial Phase")
    plt.ylabel("Number of Clinical Trials")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "phase_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Clinical trial phase distribution plot saved successfully.")

def plot_sex_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of participant sex eligibility.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        None
    """

    print("=" * 60)
    print("Sex Distribution")
    print("=" * 60)

    sex_counts = (df["sex"].fillna("Not Specified").value_counts())

    plt.figure(figsize=(12, 6))

    sex_counts.plot(kind="bar")

    plt.title("Participant Sex Distribution")

    plt.xlabel("Sex Eligibility")
    plt.ylabel("Number of Clinical Trials")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "sex_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Sex distribution plot saved successfully.")

def plot_summary_length_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of clinical trial summary lengths.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        None
    """

    print("=" * 60)
    print("Clinical Trial Summary Length Distribution")
    print("=" * 60)

    summary_length = (df["brief_summary"].str.split().str.len())

    plt.figure(figsize=(12, 6))

    plt.hist(summary_length, bins=30)

    plt.title("Clinical Trial Summary Length Distribution")

    plt.xlabel("Number of Words")

    plt.ylabel("Number of Clinical Trials")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "summary_length_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Summary length distribution plot saved successfully.")

def plot_top_words(df: pd.DataFrame) -> None:
    """
    Plot the top 20 most frequent words in clinical trial summaries.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        None
    """

    print("=" * 60)
    print("Top 20 Most Frequent Words")
    print("=" * 60)

    # Combine all summaries into one string
    text = " ".join(df["brief_summary"])

    # Split into words
    words = text.split()

    # Count word frequencies
    word_counts = Counter(words).most_common(20)

    # Convert to DataFrame
    word_df = pd.DataFrame(word_counts,columns=["Word", "Frequency"])

    plt.figure(figsize=(12, 6))

    plt.bar(word_df["Word"],word_df["Frequency"])

    plt.title("Top 20 Most Frequent Words")

    plt.xlabel("Words")
    plt.ylabel("Frequency")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "top_words.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Top words plot saved successfully.")

def generate_wordcloud(df: pd.DataFrame) -> None:
    """
    Generate a word cloud from clinical trial summaries.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        None
    """

    print("=" * 60)
    print("Generating Word Cloud")
    print("=" * 60)

    # Combine all summaries into one string
    text = " ".join(df["brief_summary"])

    # Generate word cloud
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    ).generate(text)

    plt.figure(figsize=(15, 8))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.title("Word Cloud of Clinical Trial Summaries")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "wordcloud.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Word cloud generated successfully.")

def perform_eda() -> None:
    """
    Execute the complete exploratory data analysis workflow.

    Returns:
        None
    """

    df = load_cleaned_dataset()

    dataset_summary(df)

    plot_disease_distribution(df)

    plot_study_status_distribution(df)

    plot_study_type_distribution(df)

    plot_phase_distribution(df)

    plot_sex_distribution(df)

    plot_summary_length_distribution(df)

    plot_top_words(df)

    generate_wordcloud(df)

if __name__ == "__main__":
    perform_eda()
