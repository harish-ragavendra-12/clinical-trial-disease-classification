import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.config import (
    MODELS_DIR,
    BEST_MODEL_FILE,
    VECTORIZER_FILE
)

from src.feature_engineering import feature_engineering


def load_feature_engineered_data():
    """
    Load feature engineered data.

    Returns:
        tuple:
            X : Text features
            y : Encoded target labels
    """

    return feature_engineering()


def split_dataset(X, y):
    """
    Split dataset into training and testing sets.

    Parameters:
        X: Text features
        y: Encoded target labels

    Returns:
        X_train, X_test, y_train, y_test
    """

    print("=" * 60)
    print("Splitting Dataset")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test):
    """
    Convert text data into TF-IDF vectors.

    Returns:
        X_train_vectorized,
        X_test_vectorized,
        vectorizer
    """

    print("=" * 60)
    print("Vectorizing Text using TF-IDF")
    print("=" * 60)

    vectorizer = TfidfVectorizer()

    X_train_vectorized = vectorizer.fit_transform(
        X_train
    )

    X_test_vectorized = vectorizer.transform(
        X_test
    )

    print(
        f"Training Feature Matrix Shape: {X_train_vectorized.shape}"
    )

    print(
        f"Testing Feature Matrix Shape : {X_test_vectorized.shape}"
    )

    return (
        X_train_vectorized,
        X_test_vectorized,
        vectorizer
    )


def save_vectorizer(vectorizer):
    """
    Save TF-IDF vectorizer.
    """

    print("=" * 60)
    print("Saving TF-IDF Vectorizer")
    print("=" * 60)

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE
    )

    print(
        f"Vectorizer saved successfully:\n{VECTORIZER_FILE}"
    )


def train_logistic_regression(
        X_train_vectorized,
        y_train
):
    """
    Train Logistic Regression model.
    """

    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    model.fit(
        X_train_vectorized,
        y_train
    )

    return model


def train_multinomial_naive_bayes(
        X_train_vectorized,
        y_train
):
    """
    Train Multinomial Naive Bayes model.
    """

    model = MultinomialNB()

    model.fit(
        X_train_vectorized,
        y_train
    )

    return model


def train_random_forest(
        X_train_vectorized,
        y_train
):
    """
    Train Random Forest model.
    """

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(
        X_train_vectorized,
        y_train
    )

    return model


def train_linear_svc(
        X_train_vectorized,
        y_train
):
    """
    Train Linear SVC model.
    """

    model = LinearSVC(
        random_state=42
    )

    model.fit(
        X_train_vectorized,
        y_train
    )

    return model


def evaluate_model(
        model,
        X_test_vectorized,
        y_test
):
    """
    Evaluate model performance.
    """

    y_prediction = model.predict(
        X_test_vectorized
    )

    accuracy = accuracy_score(
        y_test,
        y_prediction
    )

    report = classification_report(
        y_test,
        y_prediction
    )

    matrix = confusion_matrix(
        y_test,
        y_prediction
    )

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print("\nClassification Report")
    print(report)

    print("\nConfusion Matrix")
    print(matrix)

    return accuracy, report, matrix


def compare_models(
        X_train_vectorized,
        X_test_vectorized,
        y_train,
        y_test
):
    """
    Train and compare multiple models.

    Returns:
        best_model,
        best_model_name,
        best_accuracy
    """

    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)


    models = {

        "Logistic Regression":
            train_logistic_regression(
                X_train_vectorized,
                y_train
            ),

        "Multinomial Naive Bayes":
            train_multinomial_naive_bayes(
                X_train_vectorized,
                y_train
            ),

        "Random Forest":
            train_random_forest(
                X_train_vectorized,
                y_train
            ),

        "Linear SVC":
            train_linear_svc(
                X_train_vectorized,
                y_train
            )
    }


    accuracies = {}


    for name, model in models.items():

        print("\n")
        print(f"Evaluating {name}")

        accuracy, _, _ = evaluate_model(
            model,
            X_test_vectorized,
            y_test
        )

        accuracies[name] = accuracy


    print("=" * 60)
    print("Accuracy Comparison")
    print("=" * 60)


    for name, accuracy in accuracies.items():

        print(
            f"{name:<30}: {accuracy:.4f}"
        )


    best_model_name = max(
        accuracies,
        key=accuracies.get
    )


    best_model = models[
        best_model_name
    ]

    best_accuracy = accuracies[
        best_model_name
    ]


    print("=" * 60)
    print("Best Model")
    print("=" * 60)

    print(
        f"Model Name : {best_model_name}"
    )

    print(
        f"Accuracy   : {best_accuracy:.4f}"
    )


    return (
        best_model,
        best_model_name,
        best_accuracy
    )


def save_best_model(best_model):
    """
    Save trained model.
    """

    print("=" * 60)
    print("Saving Best Model")
    print("=" * 60)


    joblib.dump(
        best_model,
        BEST_MODEL_FILE
    )


    print(
        f"Best model saved successfully:\n{BEST_MODEL_FILE}"
    )


def model_training():
    """
    Execute complete training pipeline.
    """

    print("=" * 60)
    print("Starting Model Training Pipeline")
    print("=" * 60)


    # Feature engineering
    X, y = load_feature_engineered_data()


    # Train test split
    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )


    # TF-IDF vectorization
    X_train_vectorized, X_test_vectorized, vectorizer = vectorize_text(
        X_train,
        X_test
    )


    # Save vectorizer
    save_vectorizer(vectorizer)


    # Compare models
    best_model, best_model_name, best_accuracy = compare_models(
        X_train_vectorized,
        X_test_vectorized,
        y_train,
        y_test
    )


    # Save best model
    save_best_model(best_model)


    print(
        "\nModel Training Pipeline Completed Successfully."
    )


    return (
        best_model,
        best_model_name,
        best_accuracy
    )


if __name__ == "__main__":
    model_training()
