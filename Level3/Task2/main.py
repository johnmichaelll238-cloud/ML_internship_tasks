from src.data_loader import load_mnist_data
from src.preprocessor import preprocess_data
from src.model import build_model
from src.trainer import compile_model, train_models, save_model
from src.evaluation import (
    evaluate_model,
    predict,
    get_correct_predictions,
    get_misclassified_predictions
)
from src.visual import (
    plot_sample_images,
    plot_learning_curves,
    plot_correct_predictions,
    plot_misclassified_predictions
)


def main():

    # --------------------------------------------------
    # Load Data
    # --------------------------------------------------
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = load_mnist_data()

    # --------------------------------------------------
    # Visualize Sample Images
    # --------------------------------------------------
    plot_sample_images(X_train_raw, y_train_raw)

    # --------------------------------------------------
    # Preprocess Data
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = preprocess_data(
        X_train_raw,
        X_test_raw,
        y_train_raw,
        y_test_raw
    )

    # --------------------------------------------------
    # Build & Compile Model
    # --------------------------------------------------
    model = build_model()

    model.summary()

    compile_model(model)

    # --------------------------------------------------
    # Train Model
    # --------------------------------------------------
    history = train_models(
        model,
        X_train,
        y_train
    )

    # --------------------------------------------------
    # Evaluate Model
    # --------------------------------------------------
    test_loss, test_accuracy = evaluate_model(
        model,
        X_test,
        y_test
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------
    predicted_labels = predict(
        model,
        X_test
    )

    correct = get_correct_predictions(
        predicted_labels,
        y_test
    )

    incorrect = get_misclassified_predictions(
        predicted_labels,
        y_test
    )

    # --------------------------------------------------
    # Visualizations
    # --------------------------------------------------
    plot_learning_curves(history)

    plot_correct_predictions(
        X_test_raw,
        y_test,
        predicted_labels,
        correct
    )

    plot_misclassified_predictions(
        X_test_raw,
        y_test,
        predicted_labels,
        incorrect
    )

    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------
    save_model(model)

    print("\nModel saved successfully!")


if __name__ == "__main__":
    main()
