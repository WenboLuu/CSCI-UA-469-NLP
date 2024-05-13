from collections import Counter

import matplotlib.pyplot as plt

from utils import id2label


def plot_label_distribution(labels):
    """
    Plots the distribution of labels in the dataset.

    Args:
        labels (list): List of labels.
    """

    # Count the number of occurrences of each label
    label_counts = Counter(labels)

    # Extract the labels and their counts
    labels = list(label_counts.keys())
    labels.sort(key=lambda x: label_counts[x], reverse=True)
    label_counts = [label_counts[label] for label in labels]

    labels = [id2label[label] for label in labels]

    plt.figure(figsize=(10, 8))
    plt.bar(labels, label_counts)
    plt.xlabel("Labels")
    plt.ylabel("Counts (Log Scale)")
    plt.title("Label Distribution on Log Scale")

    # Set y-axis to log scale
    plt.yscale("log")
    plt.xticks(rotation=45)
    plt.show()


def plot_loss_metrics(train_losses, val_losses, precision_scores, recall_scores, f5_scores):
    """
    Plots the training and validation losses, as well as precision, recall, and F5 scores.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        precision_scores (list): List of precision scores.
        recall_scores (list): List of recall scores.
        f5_scores (list): List of F5 scores.
    """

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(10, 5))

    # Plot for losses
    color = "tab:blue"
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(train_losses, label="Train Loss", color="tab:red")
    ax1.plot(val_losses, label="Validation Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.legend(loc="upper left")
    ax1.set_title("Training and Validation Loss")

    # Plot for precision and recall
    color = "tab:green"
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Scores", color=color)
    ax2.plot(precision_scores, label="Precision", color="tab:grey")
    ax2.plot(recall_scores, label="Recall", color="tab:red")
    ax2.plot(f5_scores, label="F5", color="tab:green")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.legend(loc="upper left")
    ax2.set_title("Metrics")

    fig.tight_layout()  # Adjust the layout to make room for all elements
    plt.show()
