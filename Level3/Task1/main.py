from src.data_loader import load_data, get_counts
from src.visual import plot_class_distribution
from src.trainer import train_model
from src.model import save_model
from src.evaluation import evaluate_model

def main():
     # Load and preprocess
    ...
    print("=" * 50)
    print("Sentiment Classification using Linear SVM")
    print("=" * 50)

    # Load and preprocess data
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_data()
    
    class_counts = get_counts()

    # Visualize
    ...
    plot_class_distribution(class_counts)
    # Train Linear
    ...
    linear_SVM, vectorizer = train_model(X_train, y_train)

   
    # Evaluate Linear
    ...
    X_test_tfidf =  vectorizer.transform(X_test)
    
    (
        accuracy,
        classification,
        cm_shape
    )  = evaluate_model(linear_SVM, X_test_tfidf, y_test)
   
    # Serialize
    ...
    save_model(linear_SVM, vectorizer)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
    