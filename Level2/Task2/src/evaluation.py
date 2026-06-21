from sklearn.metrics import accuracy_score


def evaluate_candidate_models(
    candidate_models,
    ccp_alphas,
    X_train,
    X_test,
    y_train,
    y_test,
) -> list[dict]:
    """
    Evaluates individual models in the list containing the candidate models
    and returns a list of dictionaries. Each dictionary is a separate evaluation report
    """
    results = []
    for alpha, model in zip(ccp_alphas, candidate_models):
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        experiment = {
            "ccp_alpha": alpha,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "model": model,
        }

        results.append(experiment)

    return results


def select_best_model(evaluation_results):
    """Return the result dict with the highest test accuracy."""
    if not evaluation_results:
        raise ValueError("evaluation_results is empty")

    return max(evaluation_results, key=lambda result: result["test_accuracy"])

def select_best_model(evaluation_results):
    """
    Selects best performing model based on test set performance and accuracy
    RETURNS:
    dict: model and its metrics
    """

    if not evaluation_results:
        raise ValueError("evaluation_results is empty")

    best_result = max(
        evaluation_results,
        key=lambda result: result["test_accuracy"]
    )

    return best_result
