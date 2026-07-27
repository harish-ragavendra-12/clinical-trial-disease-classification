from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model Directory
MODELS_DIR = PROJECT_ROOT / "models"

# Figures Directory
FIGURES_DIR = PROJECT_ROOT / "figures"

# Dataset Files
RAW_DATA_FILE = RAW_DATA_DIR / "clinical_trials_raw_patient2trial_conditions.csv"

PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "clinical_trials_processed.csv"

CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "clinical_trials_cleaned.csv"