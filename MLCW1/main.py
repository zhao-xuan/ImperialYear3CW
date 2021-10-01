import numpy as np
import math

# learn and build the decision tree
def decision_tree_learning(training_dataset, depth):
    print("depth at" + str(depth))
    label = np.transpose(training_dataset)[-1]
    pivot = label[0]
    labels = [i == pivot for i in label]
    if all(labels):
        return ({'room': pivot}, depth)
    else:
        (attr, val, l_dataset, r_dataset) = find_split(training_dataset)
        node = {'attr': attr, 'value': val, 'left': None, 'right': None}
        (left, l_depth) = decision_tree_learning(l_dataset, depth + 1) if l_dataset is not None else (None, depth)
        (right, r_depth) = decision_tree_learning(r_dataset, depth + 1) if r_dataset is not None else (None, depth)
        node['left'], node['right'] = left, right
        return (node, max(l_depth, r_depth))

# find the best split among the contiguous values that have the highest information gain
# first sort the values of the attribute and then split only points between two examples in sorted order
def find_split(training_dataset):
    [input, label] = np.hsplit(training_dataset, [-1])
    attribute = val = l_dataset = r_dataset = None
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
                (S_left if input[j][attr] <= split_point else S_right).append(np.concatenate((input[j], label[j])))
            S_left = np.array(S_left)
            S_right = np.array(S_right)
            if attribute == None or (val != None and information_gain(training_dataset, S_left, S_right) > val):
                attribute, val = i, split_point
                l_dataset, r_dataset = S_left, S_right

    return (attribute, val, l_dataset, r_dataset)

# calculate information gain
# S_all is a tuple while S_left and S_right are matrices
def information_gain(S_all, S_left, S_right):
    def entropy(label):
        entropy = 0
        for i in range(1,5):
            p = len(list(filter(lambda x: x == i, label))) / len(label)
            entropy += -p * math.log(p, 2)
        return entropy

    label = np.hsplit(S_all, [-1])[1]
    num_data = len(S_all)
    all_entropy = entropy(label)
    left_entropy, right_entropy = entropy(np.hsplit(S_left, [-1])[1]), entropy(np.hsplit(S_right, [-1])[1])
    remainder = len(S_left) / num_data * left_entropy + len(S_right) / num_data * right_entropy

    return all_entropy - remainder

# print/visualize the tree
def print_decision_tree(root):
    return

# evaluating decision tree using a test dataset
def evaluate_decision_tree(test_dataset, trained_tree):
    confusion_matrix = [[0, 0, 0, 0] for _ in range(4)]
    for i in test_dataset:
        node = trained_tree[0]
        while 'room' not in node.keys():
            node = node['left'][0] if i[node['attr']] <= node['value'] else node['right'][0]
        confusion_matrix[test_dataset[-1]][node['room']] += 1
    
    return confusion_matrix

# cross-validation evaluation
def cross_validation_evaluation(dataset, depth):
    proportion = len(dataset) // 10
    cross_val_confusmat = []
    for i in range(10):
        print("cross validation fold " + str(i))
        front, back = dataset[:i*proportion] if i != 0 else [], dataset[(i+1)*proportion:] if i != 9 else []
        training_set = back if front == [] else (front if back == [] else front + back)
        test_set = dataset[i*proportion:(i+1)*proportion]
        trained_tree = decision_tree_learning(training_set, depth)
        cross_val_confusmat += evaluate_decision_tree(test_set, trained_tree)
    
    return cross_val_confusmat

# pruning
def decision_tree_pruning():
    return

# main function, including cross-validation and metric calculation
data = np.loadtxt("./wifi_db/clean_dataset.txt")
cross_validation_evaluation(data, 0)