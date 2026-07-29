import joblib
import re
import pandas as pd
from fontTools.misc import transform
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import BEST_MODEL_FILE, VECTORIZER_FILE, LABEL_ENCODER_FILE


def load_artifacts():
    """
    Load the trained model, TF-IDF vectorizer, and label encoder.

    Returns:
        tuple:
            best_model: Trained machine learning model.
            vectorizer: Fitted TF-IDF vectorizer.
            label_encoder: Fitted label encoder.
    """

    print("=" * 60)
    print("Loading Model Artifacts")
    print("=" * 60)

    best_model = joblib.load(BEST_MODEL_FILE)

    vectorizer = joblib.load(VECTORIZER_FILE)

    label_encoder = joblib.load(LABEL_ENCODER_FILE)

    print("Artifacts loaded successfully.")

    return best_model, vectorizer, label_encoder

def clean_input_text(input_text: str):
    """
    Clean the user input text for disease prediction.

    Parameters:
        input_text (str): Clinical trial summary entered by the user.

    Returns:
        str: Cleaned input text.
    """

    print("=" * 60)
    print("Cleaning Input Text")
    print("=" * 60)

    #Converts text to lowercase.
    input_text = input_text.lower()

    #Removes HTML tags.
    input_text = re.sub(r"<.*?>", "", input_text)

    #Removes URLs.
    input_text = re.sub(r"http\S+|www\S+", "", input_text)

    #Removes punctuation/special characters.
    input_text = re.sub(r"[^\w\s]", "", input_text)

    #Removes extra whitespace.
    input_text = re.sub(r"\s+", " ", input_text)

    #Returns the cleaned string.
    input_text = input_text.strip()

    print("Input text cleaned successfully.")

    return input_text

def vectorize_input_text(cleaned_input_text: str, vectorizer):
    """
    Convert the cleaned input text into TF-IDF feature vectors.

    Parameters:
        cleaned_input_text (str): Cleaned clinical trial summary.
        vectorizer: Fitted TF-IDF vectorizer.

    Returns:
        scipy.sparse.csr_matrix: Vectorized input text.
    """

    print("=" * 60)
    print("Vectorizing Input Text")
    print("=" * 60)

    vectorized_text = vectorizer.transform([cleaned_input_text])

    print("Input text vectorized successfully.")

    return vectorized_text

def predict_disease(best_model, vectorized_text, label_encoder) -> str:
    """
    Predict the disease category from the input text.

    Parameters:
        best_model: Trained machine learning model.
        vectorized_text: TF-IDF vectorized input text.
        label_encoder: Fitted label encoder.

    Returns:
        str: Predicted disease category.
    """

    print("=" * 60)
    print("Predicting Disease")
    print("=" * 60)

    predicted_label = best_model.predict(vectorized_text)

    predicted_disease = label_encoder.inverse_transform(predicted_label)[0]

    print(f"Predicted Disease: {predicted_disease}")

    return predicted_disease

def prediction_pipeline(input_text: str) -> str:
    """
    Execute the complete disease prediction pipeline.

    Parameters:
        input_text (str): Clinical trial summary entered by the user.

    Returns:
        str: Predicted disease category.
    """

    print("=" * 60)
    print("Starting Prediction Pipeline")
    print("=" * 60)

    best_model, vectorizer, label_encoder = load_artifacts()

    cleaned_input_text = clean_input_text(input_text)

    vectorized_text = vectorize_input_text(
        cleaned_input_text,
        vectorizer
    )

    predicted_disease = predict_disease(
        best_model,
        vectorized_text,
        label_encoder
    )

    print("\nPrediction Pipeline Completed Successfully.")

    return predicted_disease

if __name__ == "__main__":
    input_text = (
        "A randomized clinical trial evaluating the effectiveness "
        "of insulin therapy in patients with type 2 diabetes."
    )

    predicted_disease = prediction_pipeline(input_text)

    print(f"\nPredicted Disease: {predicted_disease}")
