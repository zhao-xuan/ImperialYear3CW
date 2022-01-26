# A collection of functions to compute and present metrics

import numpy as np
from utils import COLORS, BOLD, RESET


# A pretty-printed summary of the computed pruned and unpruned tree metrics
def summary(metrics_unpruned, metrics_pruned):
    labels = [
        "Recall (Per Class)",
        "Precision (Per Class)",
        "Macro-averaged F1 Score",
        "Accuracy (Average)",
    ]

    # Round off the metrics
    # Note: The reason we split it is because the data is of different shapes!
    #       Recall and precisions are an array of values (one for each class)
    metrics_unpruned[:2] = np.round(metrics_unpruned[:2], decimals=3)
    metrics_pruned[:2] = np.round(metrics_pruned[:2], decimals=3)

    metrics_unpruned[2:] = np.round(metrics_unpruned[2:], decimals=3)
    metrics_pruned[2:] = np.round(metrics_pruned[2:], decimals=3)

    # Pretty-print the metrics section
    print(COLORS[2] + BOLD + "Metrics".ljust(80, "─") + RESET)
    for i, label in enumerate(labels):
        print(
            f"{label:<24} Unpruned: {metrics_unpruned[i]}\n"
            f"{'':<24} Pruned  : {metrics_pruned[i]}\n"
            f"{'─' * 80 * (1 - i // len(labels))}"  # No line after last metric
        )


# Computes the accuracy of the decision tree, given its confusion matrix
def accuracy(matrix):
    return np.sum(np.diagonal(matrix)) / np.sum(matrix)


# Computes the recall of the decision tree, given its confusion matrix
def recall(matrix):
    return np.diagonal(matrix) / np.sum(matrix, axis=1)


# Computes the precision of the decision tree, given its confusion matrix
def precision(matrix):
    return np.diagonal(matrix) / np.sum(matrix, axis=0)


# Computes the macro-averaged F1 score of the decision tree, given its confusion
# matrix; note that this function invokes precision and recall.
def f1(matrix):
    avg_precision = np.average(precision(matrix))
    avg_recall = np.average(recall(matrix))
    return (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)
