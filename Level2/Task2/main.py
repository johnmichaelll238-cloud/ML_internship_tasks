from src.config import DATA_PATH, MODEL_FILENAME, TREE_IMAGE_FILENAME

from src.data_loader import load_data
from src.trainer import (
    train_baseline,
    generate_pruning_path,
    train_candidate_models,
    train_final_model
)
from src.evaluation import (
    evaluate_candidate_models,
    select_best_model
)
from src.visual import plot_decision_tree

import joblib


def main():
    # 1. Load data
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    # 2. Train baseline model
    baseline_model = train_baseline(X_train, y_train)

    # 3. Get pruning path
    ccp_alphas = generate_pruning_path(
        baseline_model, X_train, y_train
    )

    # 4. Train candidate models
    candidate_models = train_candidate_models(
        X_train, y_train, ccp_alphas
    )

    # 5. Evaluate models
    results = evaluate_candidate_models(
        candidate_models,
        ccp_alphas,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # 6. Select best model
    best_result = select_best_model(results)
    best_model = best_result["model"]

    # 7. Save model
    joblib.dump(best_model, MODEL_FILENAME)

    # 8. Visualize tree
    plot_decision_tree(best_model, X_train.columns, TREE_IMAGE_FILENAME)

    print("Pipeline executed successfully.")
    print(f"Best Test Accuracy: {best_result['test_accuracy']:.4f}")
    
    

if __name__ == "__main__":
    main()