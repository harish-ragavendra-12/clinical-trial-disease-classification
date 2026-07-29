from pathlib import Path


# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================================
# Directory Paths
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

FIGURES_DIR = PROJECT_ROOT / "figures"


# ==========================================================
# Raw Dataset
# ==========================================================

RAW_DATA_FILE = (
    RAW_DATA_DIR /
    "clinical_trials_raw_patient2trial_conditions.csv"
)


# ==========================================================
# Processed Datasets
# ==========================================================

PROCESSED_DATA_FILE = (
    PROCESSED_DATA_DIR /
    "clinical_trials_processed.csv"
)

CLEANED_DATA_FILE = (
    PROCESSED_DATA_DIR /
    "clinical_trials_cleaned.csv"
)

FEATURE_ENGINEERED_DATA_FILE = (
    PROCESSED_DATA_DIR /
    "feature_engineered_clinical_trials.csv"
)


# ==========================================================
# Model Artifacts
# ==========================================================

BEST_MODEL_FILE = (
    MODELS_DIR /
    "best_model.joblib"
)

VECTORIZER_FILE = (
    MODELS_DIR /
    "tfidf_vectorizer.joblib"
)

LABEL_ENCODER_FILE = (
    MODELS_DIR /
    "label_encoder.joblib"
)