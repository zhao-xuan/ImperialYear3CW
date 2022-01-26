# A decision tree algorithm that determines a user's location (i.e. room) based
# on WiFi signal strengths collected from a mobile phone.

import utils
import metrics
import numpy as np
import functools
import sys

from multiprocessing import Pool
from utils import COLORS, BOLD, RESET


FOLD = 10  # The number of folds for cross validation
PROP = 0.1  # The proportion of samples to use for the test datset
SEED = 60012  # The seed to a random number generator

# Paths to the datasets
CLEAN_PATH = "./wifi_db/clean_dataset.txt"  # Relative path to the clean dataset
NOISY_PATH = "./wifi_db/noisy_dataset.txt"  # Relative path to the noisy dataset


# Note on the dataset:
# Each sample contains 7 WiFi signal strenths and the number of the room where
# the user was standing when these signal strengths were recorded.
# The signal strenths are continuous; the room number is discrete.


def main(data_set, is_pruned):
    # Select dataset path and print dataset choice to terminal
    path = CLEAN_PATH if data_set == "clean" else NOISY_PATH
    placeholder = "noisy" if path == NOISY_PATH else "clean"
    print(COLORS[1] + BOLD + f"Note: Using {placeholder} dataset" + RESET)

    # Load the data from the chosen dataset and shuffle its contents; a seed is
    # setup for experiment reproducibility reasons
    data = np.loadtxt(path)
    np.random.seed(SEED)
    np.random.shuffle(data)

    # Make sure that the data wasn't accidentally modified
    assert data.shape == (2000, 8)

    # Compute the pruned and unpruned confusion matrices for each fold of the
    # 10-fold cross validation process; note that the computation is run via a
    # pool of worker threads for vastly improved performance.
    confusion_matrices = []
    with Pool(FOLD) as pool:
        # A partial function is used to pre-fill the data argument
        f = functools.partial(cross_validation, data, is_pruned)

        # Launch worker threads and collect their results in confusion_matrices
        confusion_matrices = pool.map(f, range(FOLD))

    # Collect the unpruned confusion matrices in a list at index 0 and the
    # pruned confusion matrices at index 1.
    # Note: The * operator unpacks the list of confusion matrices from each fold
    #       The zip returns an iterator so we wrap it in list
    confusion_matrices = np.array(list(zip(*confusion_matrices)))
    unpruned_confusion = utils.sum_matrices(confusion_matrices[0])
    pruned_confusion = utils.sum_matrices(confusion_matrices[1])

    # Results:
    # Display the computed cumulative unpruned and pruned confusion matrices
    print(COLORS[2] + BOLD + "Confusion Matrices ".ljust(80, "─") + RESET)
    print(f"Cumulative Unpruned Confusion Matrix:\n{unpruned_confusion}\n")
    print(f"Cumulative Pruned Confusion Matrix:\n{pruned_confusion}\n")

    # Compute the metrics
    fs = [metrics.recall, metrics.precision, metrics.f1, metrics.accuracy]
    metrics_unpruned = [f(unpruned_confusion) for f in fs]
    metrics_pruned = [f(pruned_confusion) for f in fs]

    # A summary of the calculated statistics
    metrics.summary(metrics_unpruned, metrics_pruned)


# Compute the confusion matrices of the unpruned and pruned decision trees.
# Note: The unpruned tree is generated using the train and test sets, which are
#       computed by splitting the dataset at the provided fold number.
#       A validation set used to prune the unpruned tree.
def cross_validation(data, is_pruned, fold_number):
    fold_size = int(len(data) * PROP)
    val_index = (fold_number + 1) % FOLD  # The index of the validation set

    test_slice = np.s_[fold_number * fold_size : (fold_number + 1) * fold_size]
    val_slice = np.s_[val_index * fold_size : (val_index + 1) * fold_size]

    # Compute the test, validation, and training sets using the slices
    test_set = data[test_slice]
    val_set = data[val_slice]
    train_set = np.delete(data, np.r_[test_slice, val_slice], axis=0)

    # Train and evaluate the unpruned tree; then prune the tree using the
    # validation set and evaluate it against the test set.
    (unpruned_tree, _) = decision_tree_learning(train_set, 0)
    unpruned_confusion = evaluate_decision_tree(test_set, unpruned_tree)

    # Print a visualisation of the unpruned decision tree
    if fold_number == 0 and is_pruned == "unpruned":
        print(COLORS[2] + BOLD + "Visualisation ".ljust(80, "─") + RESET)
        utils.visualise_tree(unpruned_tree)
        print(f"\nAverage depth: {utils.average_depth(unpruned_tree)}\n")

    pruned_tree = decision_tree_pruning(val_set, unpruned_tree)
    pruned_confusion = evaluate_decision_tree(test_set, pruned_tree)

    # Print a visualisation of the pruned decision tree
    if fold_number == 0 and is_pruned == "pruned":
        print(COLORS[2] + BOLD + "Visualisation ".ljust(80, "─") + RESET)
        utils.visualise_tree(pruned_tree)
        print(f"\nAverage depth: {utils.average_depth(pruned_tree)}\n")

    return (unpruned_confusion, pruned_confusion)


# Learn and build the decision tree
def decision_tree_learning(train_set, depth):
    labels = np.transpose(train_set)[-1]  # Get all the rooms

    # If all the labels are the same, then create and return a leaf node
    if all([i == labels[0] for i in labels]):
        return ({"room": labels[0], "count": len(labels)}, depth)

    # Compute the split with the highest information gain and create a new
    # decision tree node with its value as the split point
    (attr, val, l_set, r_set) = utils.find_split(train_set)
    node = {"attr": attr, "value": val, "left": None, "right": None}

    (left, l_depth) = (right, r_depth) = (None, depth)
    if l_set is not None:
        (left, l_depth) = decision_tree_learning(l_set, depth + 1)
    if r_set is not None:
        (right, r_depth) = decision_tree_learning(r_set, depth + 1)

    node["left"], node["right"] = left, right
    return (node, max(l_depth, r_depth))


# Evaluate decision tree against the test dataset
def evaluate_decision_tree(test_set, root):
    confusion_matrix = np.zeros((4, 4))  # Initialise the confusion matrix
    for test in test_set:
        node = root

        # Determine whether we hit a leaf node in the decision tree; if not,
        # walk down the branches of the tree.
        while "room" not in node.keys():
            if test[node["attr"]] <= node["value"]:
                node = node["left"]
            else:
                node = node["right"]

        true_room = int(test[-1] - 1)  # The true/gold room
        pred_room = int(node["room"] - 1)  # The room predicted by the tree
        confusion_matrix[true_room][pred_room] += 1

    return confusion_matrix


# Recursively prune the decision tree against the given validation set
def decision_tree_pruning(validation_set, root):
    if utils.is_leaf(root):
        return root

    # Prune the left and right branches from the given root recursively against
    # a subset of the validation set.
    root["left"] = decision_tree_pruning(
        validation_set[np.where(validation_set[:, root["attr"]] <= root["value"])],
        root["left"],
    )
    root["right"] = decision_tree_pruning(
        validation_set[np.where(validation_set[:, root["attr"]] > root["value"])],
        root["right"],
    )

    # Case where we encounter a twig node (left and right are leaves):
    if utils.is_leaf(root["left"]) and utils.is_leaf(root["right"]):
        if root["left"]["room"] == root["right"]["room"]:
            root["left"]["count"] += root["right"]["count"]
            return root["left"]

        majority_node = max(root["left"], root["right"], key=lambda n: n["count"])
        if validation_set.size == 0:
            return majority_node

        # Prune only if there's an improvement in accuracy
        after_prune = evaluate_decision_tree(validation_set, majority_node)
        before_prune = evaluate_decision_tree(validation_set, root)

        if metrics.accuracy(after_prune) >= metrics.accuracy(before_prune):
            return majority_node

    return root


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 main.py [data_set] [is_pruned]")
        print("\t[data_set] is the name of the data set to use: clean or noisy")
        print("\t[is_pruned] indicates whether to print pruned tree: pruned or unpruned")
        sys.exit(1)
    data_set = sys.argv[1]
    is_pruned = sys.argv[2]
    if data_set not in ["train", "test"]:
        print("Please specify the data set to use: train or test")
    if is_pruned not in ["pruned", "unpruned"]:
        print("Please specify whether to use the pruned tree")
    main(data_set, is_pruned)