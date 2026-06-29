from sklearn.metrics import(
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model, X_test_tfidf, y_test):
    y_pred_linear = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred_linear)
    classification = classification_report(y_test, y_pred_linear)
    cm = confusion_matrix(y_test, y_pred_linear)
    #remember to return cm.shape
    return (
        accuracy,
        classification,
        cm.shape
    )
    