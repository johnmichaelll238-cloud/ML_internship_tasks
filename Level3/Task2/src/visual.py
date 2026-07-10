import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_sample_images(images, labels):
    """
    Display a 5x5 grid of sample MNIST images.
    """

    plt.figure(figsize=(8, 8))

    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(labels[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_learning_curves(history):
    """
    Plot training and validation loss/accuracy.
    """

    metrics_df = pd.DataFrame(history.history)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics_df["loss"], label="Training Loss")
    plt.plot(metrics_df["val_loss"], label="Validation Loss")
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics_df["accuracy"], label="Training Accuracy")
    plt.plot(metrics_df["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_correct_predictions(
    images,
    labels,
    predicted_labels,
    correct_indices,
    num_images=9
):
    """
    Display randomly selected correctly classified images.
    """

    selected = np.random.choice(
        correct_indices,
        size=min(num_images, len(correct_indices)),
        replace=False
    )

    plt.figure(figsize=(8, 8))

    for i, idx in enumerate(selected):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[idx], cmap="gray")
        plt.title(
            f"True: {labels[idx]}\nPred: {predicted_labels[idx]}"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_misclassified_predictions(
    images,
    labels,
    predicted_labels,
    incorrect_indices,
    num_images=9
):
    """
    Display randomly selected misclassified images.
    """

    selected = np.random.choice(
        incorrect_indices,
        size=min(num_images, len(incorrect_indices)),
        replace=False
    )

    plt.figure(figsize=(8, 8))

    for i, idx in enumerate(selected):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[idx], cmap="gray")
        plt.title(
            f"True: {labels[idx]}\nPred: {predicted_labels[idx]}"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()
