from src.evaluation import evaluate_model


# ==========================================================
# Compare Models
# ==========================================================

def compare_models(
        logistic_model,
        mnb_model,
        random_forest_model,
        linear_svc_model,
        X_test_vectorized,
        y_test
):
    """
    Compare trained models and return the best model.
    """

    print("=" * 60)
    print("Comparing Models")
    print("=" * 60)

    logistic_accuracy, _, _ = evaluate_model(
        logistic_model,
        X_test_vectorized,
        y_test
    )

    mnb_accuracy, _, _ = evaluate_model(
        mnb_model,
        X_test_vectorized,
        y_test
    )

    random_forest_accuracy, _, _ = evaluate_model(
        random_forest_model,
        X_test_vectorized,
        y_test
    )

    linear_svc_accuracy, _, _ = evaluate_model(
        linear_svc_model,
        X_test_vectorized,
        y_test
    )

    model_scores = {
        "Logistic Regression": logistic_accuracy,
        "Multinomial Naive Bayes": mnb_accuracy,
        "Random Forest": random_forest_accuracy,
        "Linear SVC": linear_svc_accuracy
    }

    print("\nModel Performance")

    for model_name, accuracy in model_scores.items():

        print(f"{model_name:<30}: {accuracy:.4f}")

    best_model_name = max(
        model_scores,
        key=model_scores.get
    )

    print("\nBest Model :", best_model_name)

    if best_model_name == "Logistic Regression":
        return logistic_model

    elif best_model_name == "Multinomial Naive Bayes":
        return mnb_model

    elif best_model_name == "Random Forest":
        return random_forest_model

    else:
        return linear_svc_model
