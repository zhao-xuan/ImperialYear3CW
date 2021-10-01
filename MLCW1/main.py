import numpy as np
import math

# Note: all dataset variable is a two-element tuple with the format (input, label)
# learn and build the decision tree
def decision_tree_learning(training_dataset, depth):
    (_, label) = training_dataset
    pivot = label[0]
    labels = [i == pivot for i in label]
    if all(labels):
        return ({'room': pivot}, depth)
    else:
        (attr, val, l_dataset, r_dataset) = find_split(training_dataset)
        node = {'attribute': attr, 'value': val, 'left': None, 'right': None}
        (left, l_depth) = decision_tree_learning(l_dataset, depth + 1)
        (right, r_depth) = decision_tree_learning(r_dataset, depth + 1)
        node['left'], node['right'] = left, right
        return (node, max(l_depth, r_depth))

# find the best split among the contiguous values that have the highest information gain
# first sort the values of the attribute and then split only points between two examples in sorted order
def find_split(training_dataset):
    (input, label) = training_dataset
    attr = val = l_dataset = r_dataset = None
    # for every column, sort it and then split between two consecutive values
    num_column = np.shape(input)[1]
    for attr in range(num_column):
        sorted_column = sorted(np.transpose(input)[attr])
        for i in range(len(sorted_column) - 1):
            if sorted_column[i] == sorted_column[i+1]:
                continue
            split_point = (sorted_column[i] + sorted_column[i+1]) / 2
            S_left = S_right = []
            for j in range(len(input)):
                (S_left if input[j][i] <= split_point else S_right).append(input[j] + label[j])
                if (attr == None and val == None) or information_gain(training_dataset, S_left, S_right) > val:
                    attr, val = i, split_point
                    l_dataset, r_dataset = S_left, S_right
    return (attr, val, l_dataset, r_dataset)

# calculate information gain
# S_all is a tuple while S_left and S_right are matrices
def information_gain(S_all, S_left, S_right):
    def entropy(label):
        entropy = 0
        for i in range(1,5):
            p = len(list(lambda x: x == i, label)) / len(label)
            entropy += -p * math.log(p, 2)
        return entropy

    (_, label) = S_all
    num_data = len(S_all)
    all_entropy = entropy(label)
    left_entropy, right_entropy = entropy(), entropy()
    remainder = len(S_left) / num_data * left_entropy + len(S_right) / num_data * right_entropy

    return all_entropy - remainder

# print/visualize the tree
def print_decision_tree(root):
    return

# cross-validation evaluation
# returns the accuracy of the tree
def cross_validation_evaluation(test_dataset, trained_tree):
    return

# pruning
def decision_tree_pruning():
    return

# main function, including cross-validation and metric calculation
data = np.loadtxt("./clean_dataset.txt")
[input, label, _] = np.hsplit(data, [7, 8])
label = label.astype(int)