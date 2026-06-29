from src.config import IMAGES_DIR
import matplotlib.pyplot as plt

# visual.py

def plot_class_distribution(class_counts):
    """
    Plot and save the class distribution.
    """

    class_counts.sort_values(ascending=False).plot(
        kind="bar",
        figsize=(18, 6)
    )

    plt.title("Emotion Class Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.savefig(
        IMAGES_DIR / "class_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()