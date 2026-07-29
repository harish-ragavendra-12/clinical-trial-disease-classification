from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(
        model,
        X_test_vectorized,
        y_test
):
    """
    Evaluate a trained machine learning model.

    Parameters:
        model: Trained machine learning model.
        X_test_vectorized: Vectorized test features.
        y_test: True test labels.

    Returns:
        tuple:
            accuracy
            classification_report
            confusion_matrix
    """

    print("=" * 60)
    print("Evaluating Model")
    print("=" * 60)

    # Predictions
    y_pred = model.predict(
        X_test_vectorized
    )

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # Classification Report
    report = classification_report(
        y_test,
        y_pred
    )

    # Confusion Matrix
    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    print(f"Accuracy : {accuracy:.4f}")

    return (
        accuracy,
        report,
        matrix
    )
