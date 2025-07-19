import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn import tree


def aggregate_bounds(conditions, n_features):
    bounds = [(None, None)] * n_features  # (min, max) for each feature
    for feature_index, threshold_value, condition in conditions:
        if condition == 'le':
            if bounds[feature_index][1] is None or threshold_value < bounds[feature_index][1]:
                bounds[feature_index] = (bounds[feature_index][0], threshold_value)
        elif condition == 'gt':
            if bounds[feature_index][0] is None or threshold_value > bounds[feature_index][0]:
                bounds[feature_index] = (threshold_value, bounds[feature_index][1])
    return bounds

def generate_samples_from_tree(clf, num_samples=1):
    tree = clf.tree_
    feature = tree.feature
    threshold = tree.threshold

    def recurse(node, depth, path_conditions):
        if tree.children_left[node] != _tree.TREE_LEAF:
            left_child = tree.children_left[node]
            right_child = tree.children_right[node]
            feature_index = feature[node]
            threshold_value = threshold[node]

            recurse(left_child, depth + 1, path_conditions + [(feature_index, threshold_value, 'le')])
            recurse(right_child, depth + 1, path_conditions + [(feature_index, threshold_value, 'gt')])
        else:
            leaf_conditions.append(path_conditions)

    leaf_conditions = []
    recurse(0, 1, [])

    # Get global min and max for each feature from the whole tree
    global_bounds = [(float('inf'), float('-inf'))] * clf.n_features_in_
    for node in range(tree.node_count):
        if feature[node] != _tree.TREE_UNDEFINED:
            feature_index = feature[node]
            threshold_value = threshold[node]
            if threshold_value < global_bounds[feature_index][0]:
                global_bounds[feature_index] = (threshold_value, global_bounds[feature_index][1])
            if threshold_value > global_bounds[feature_index][1]:
                global_bounds[feature_index] = (global_bounds[feature_index][0], threshold_value)

    samples = []
    for conditions in leaf_conditions:
        bounds = aggregate_bounds(conditions, clf.n_features_in_)
        for i in range(clf.n_features_in_):
            if bounds[i][0] is None:
                bounds[i] = (global_bounds[i][0], bounds[i][1])
            if bounds[i][1] is None:
                bounds[i] = (bounds[i][0], global_bounds[i][1])
        for _ in range(num_samples):
            sample = np.empty(clf.n_features_in_)
            for i in range(clf.n_features_in_):
                sample[i] = np.random.uniform(low=bounds[i][0], high=bounds[i][1])
            samples.append(sample)

    return np.array(samples, dtype=object)


def plot_tree(model, x_train, target_path):
    tree.plot_tree(model, filled=True, feature_names=x_train.columns, class_names=['0', '1'])
    plt.savefig(target_path, dpi=700)
