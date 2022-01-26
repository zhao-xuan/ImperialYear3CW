# A collection of utility functions and global constants

import functools
import numpy as np

# Colours to identify branch levels of the decision tree in the terminal; note
# that the colours are cycled.
COLORS = [
    "\u001b[31m",  # Red
    "\u001b[33m",  # Yellow
    "\u001b[32m",  # Green
    "\u001b[34m",  # Blue
    "\u001b[35m",  # Magenta
    "\u001b[36m",  # Cyan
]
WHITE = "\u001b[37m"
BOLD = "\u001b[1m"
RESET = "\u001b[0m"  # Reset all text formatting to defaults


# Finds the best split among the contiguous values that have the highest
# information gain. Returns a tuple containing the attribute, value, and left
# and right subsets.
def find_split(dataset):
    columns = np.shape(dataset)[1] - 1  # The number of feature columns
    split = (None, None, None, None)
    max_gain = 0

    # For every feature (attribute) column, sort it into a row (list), split the
    # list of attribute values into two subsets, calculate the information gain
    # and update the resulting split configuration accordingly.
    for attr in range(columns):
        attr_list = sorted(np.transpose(dataset)[attr])

        for i in range(len(attr_list) - 1):
            # Zero information gain between two consecutive values, so skip
            if attr_list[i] == attr_list[i + 1]:
                continue

            # Compute the left and right subsets using the split point (val)
            val = (attr_list[i] + attr_list[i + 1]) / 2
            l_set = dataset[dataset[:, attr] <= val]
            r_set = dataset[dataset[:, attr] > val]

            # Calculate the information gain and update the result if higher
            gain = information_gain(dataset, l_set, r_set)
            if gain > max_gain:
                split = attr, val, l_set, r_set
                max_gain = gain

    return split


# Returns the information gain of the provided dataset
def information_gain(dataset, l_set, r_set):
    labels = dataset[:, -1]
    dataset_entropy = entropy(labels)
    l_entropy, r_entropy = entropy(l_set[:, -1]), entropy(r_set[:, -1])

    remainder = (len(l_set) * l_entropy + len(r_set) * r_entropy) / len(dataset)
    return dataset_entropy - remainder


# Returns the entropy of the provided class labels
def entropy(labels):
    # Get the count of each label and calculate their proportion
    _, count = np.unique(labels, return_counts=True)
    proportions = count / len(labels)
    entropy = 0

    for proportion in proportions:
        entropy -= proportion * np.log2(proportion)

    return entropy


# Return the element-wise sum of a list of matrices
def sum_matrices(matrices):
    return functools.reduce(lambda x, y: x + y, matrices)


# Check if the given node is a leaf node
def is_leaf(node):
    return "room" in node.keys()


# Pretty-print the decision tree for visualisation in the project report
def visualise_tree(node):
    print("\n".join(tree_strings(node, 0)))


# A recursive helper that generates the print strings for each level of the tree
# Note: The depth argument is used to cycle the colours for each depth.
def tree_strings(root, depth):
    if is_leaf(root):
        return [f"{WHITE}╰───leaf: {int(root['room'])}{RESET}"]

    # Step down either branch to compute strings of lower levels
    next_depth = (depth + 1) % len(COLORS)
    l_strings = tree_strings(root["left"], next_depth)
    r_strings = tree_strings(root["right"], next_depth)

    return (
        [f"{COLORS[depth]}├───X{root['attr']} <= {root['value']}{RESET}"]
        + [f"{COLORS[depth]}│   {RESET}" + s for s in l_strings]
        + [f"{COLORS[depth]}╰───X{root['attr']} > {root['value']}{RESET}"]
        + [f"{COLORS[depth]}    {RESET}" + s for s in r_strings]
    )

# Calculate the average depth of leaf nodes in the decision tree
def average_depth(tree, depth=0):
    def _average_depth(tree, depth):
        if is_leaf(tree):
            return (depth, 1)
        (left_depth, left_count) = _average_depth(tree["left"], depth + 1)
        (right_depth, right_count) = _average_depth(tree["right"], depth + 1)
        return (left_depth + right_depth, left_count + right_count)

    (depth, count) = _average_depth(tree, depth)
    return depth / count