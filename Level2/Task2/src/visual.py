from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

def plot_decision_tree(model, feature_names, output_path):
    """
    Saves a visual representation of a trained decision tree.

    Parameters:
        model: Trained DecisionTreeClassifier
        feature_names: list of feature names
        output_path: where to save the image
    """

    plt.figure(figsize=(12, 8))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=True,
        filled=True
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    