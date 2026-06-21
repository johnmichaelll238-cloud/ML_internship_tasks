from data_loader import load_data
from evaluation import evaluate_candidate_models
from trainer import train_candidate_models

X_train, X_test, y_train, y_test = load_data("iris.csv")

print(X_train.shape)
print(X_test.shape)

print(y_train.value_counts())
print(y_test.value_counts())
"""
results = evaluate_candidate_models(
    candidate_models,
    ccp_alphas,
    X_train,
    X_test,
    y_train,
    y_test
)

print(results[0])
"""