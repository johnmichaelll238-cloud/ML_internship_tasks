import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.

    Returns:
        tuple:
            (test_loss, test_accuracy)
    """

    return model.evaluate(
        X_test,
        y_test,
        verbose=1
    )
def predict(model, X_test):
    """
    Generate predictions.

    Returns:
        np.ndarray
    """

    probabilities = model.predict(X_test)

    predicted_labels = np.argmax(
        probabilities,
        axis=1
    )

    return predicted_labels

def get_correct_predictions(
    predicted_labels,
    y_test
):
    return np.where(
    predicted_labels == y_test)[0]

def get_misclassified_predictions(
    predicted_labels,
    y_test
    ):
    return np.where(
    predicted_labels != y_test)[0]
    
