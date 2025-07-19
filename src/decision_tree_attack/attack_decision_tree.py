import copy
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF

from src.decision_tree_attack.training_trees import train_decision_tree
from src.decision_tree_attack.utils import plot_tree
from src.utils.utils import read_config, path_to_config

def perform_pso_with_metadata(model: DecisionTreeClassifier, x, df):
    # Get the tree properties
    n_node_samples = model.tree_.n_node_samples
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right

    # Identify a leaf with only one sample
    single_sample_leaves = [node for node in range(len(n_node_samples)) if
                            n_node_samples[node] == 1 and
                            children_left[node] == -1 and
                            children_right[node] == -1]
    if single_sample_leaves:
        # Take the first single-sample leaf for simplicity
        leaf_node = single_sample_leaves[0]

        result = model.apply(x)
        # Find the sample index in this leaf
        leaf_sample_index = np.where(model.apply(x) == leaf_node)[0][0]

        # Extract the decision path for this sample
        node_indicator = model.decision_path(x)
        path = node_indicator.indices[
               node_indicator.indptr[leaf_sample_index]:node_indicator.indptr[leaf_sample_index + 1]]

        # Variables for storing conditions
        feature = model.tree_.feature
        threshold = model.tree_.threshold
        conditions = []

        # Construct the query string considering both directions
        for node in path[:-1]:  # Exclude the last node, which is a leaf
            if feature[node] != -2:  # Check if not a leaf node
                feature_name = df.columns[feature[node]]
                # Determine direction based on whether the left child matches the next node in the path
                if node + 1 in path and children_left[node] == node + 1:
                    condition = f"`{feature_name}` <= {threshold[node]}"
                else:
                    condition = f"`{feature_name}` > {threshold[node]}"
                conditions.append(condition)

        # Join conditions with logical 'AND'
        query_string = " & ".join(conditions)

        # Query the DataFrame
        singled_out_sample = df.query(query_string)
    else:
        singled_out_sample = "No single-sample leaves found."
        query_string = None

    return singled_out_sample, query_string


def get_so_query_values(model: Tuple[DecisionTreeClassifier, DecisionTreeRegressor], x, x_and_y):
    regression = False
    if isinstance(model, DecisionTreeRegressor):
        regression = True

    # Get the tree properties
    children_left, children_right, vulnerable_nodes, values = extract_vulnerable_nodes(model, regression)
    result = model.apply(x)
    num_so_leaves = 0

    if vulnerable_nodes:
        for node_index in range(len(vulnerable_nodes)):
            # Take the first node with a 1 for simplicity
            target_node = vulnerable_nodes[node_index]

            # Find a sample that passes through this node
            leaf_sample_index = None
            for i in range(len(result)):
                # Extract the decision path for this sample
                sample = pd.DataFrame([x.iloc[i].values], columns=x.columns)
                node_indicator = model.decision_path(sample)
                path = node_indicator.indices[
                       node_indicator.indptr[0]:node_indicator.indptr[1]]
                if target_node in path:
                    leaf_sample_index = i
                    break

            if leaf_sample_index is not None:
                # Extract the decision path for this sample
                sample = pd.DataFrame([x.iloc[leaf_sample_index].values], columns=x.columns)
                node_indicator = model.decision_path(sample)
                path = node_indicator.indices[
                       node_indicator.indptr[0]:node_indicator.indptr[1]]

            # Variables for storing conditions
            feature = model.tree_.feature
            threshold = model.tree_.threshold
            conditions = []

            # delete all nodes which come after the target node
            target_node_index = np.where(path == target_node)[0][0]

            increment = 0
            # if target node is a leaf node we do not want to add a condition
            if path[len(path) - 1] == target_node:
                increment = 1
                num_so_leaves += 1
            path = path[:target_node_index + increment]

            # Construct the query string considering both directions
            for node in path:  # Exclude the last node, which is a leaf
                if feature[node] != -2:  # Check if not a leaf node
                    feature_name = x_and_y.columns[feature[node]]
                    # Determine direction based on whether the left child matches the next node in the path
                    if node + 1 in path and children_left[node] == node + 1:
                        condition = f"`{feature_name}` <= {threshold[node]}"
                    else:
                        condition = f"`{feature_name}` > {threshold[node]}"
                    conditions.append(condition)


            # Join conditions with logical 'AND'
            query_string = " & ".join(conditions)
            # get index where leaf node has value 1
            leaf_value_index = np.where(values[target_node].flatten() == 1)[0][0]
            query_string += " & " + f"`y` == {leaf_value_index}"

            # Query the DataFrame
            singled_out_sample = x_and_y.query(query_string)
            #print(singled_out_sample["y"])
            #print(query_string)
            break
    else:
        singled_out_sample = "No single-sample leaves found."
        query_string = None

    # get number of leaf nodes in tree
    n_nodes = model.tree_.node_count
    n_leaf_nodes = sum(1 for i in range(n_nodes) if children_left[i] == children_right[i])
    n_internal_nodes = n_nodes - n_leaf_nodes
    num_so_nodes = len(vulnerable_nodes)
    print(f"Identified {num_so_nodes} singling out paths")
    print(f"Brute force SO-ACC Leaf: {num_so_leaves / n_leaf_nodes}")
    epsilon = 1e-10
    print(f"Brute force SO-ACC Nodes: {(num_so_nodes - num_so_leaves) / (n_internal_nodes + epsilon)}")

    return singled_out_sample, query_string


def extract_vulnerable_nodes(model, regression=False):
    #n_node_samples = model.tree_.n_node_samples
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    values = model.tree_.value
    samples = model.tree_.n_node_samples
    # Identify all leaf having a class distribution with at least one class with
    if not regression:
        vulnerable_nodes = [node for node in range(len(values)) if 1 in values[node]]
    else:
        vulnerable_nodes = [node for node in range(len(samples)) if 2 == samples[node]]

    #print(f"Number of vulnerable nodes: {len(vulnerable_nodes)}")
    return children_left, children_right, vulnerable_nodes, values



def perform_so_attack_without_meta_data(model: Tuple[DecisionTreeClassifier, DecisionTreeRegressor], x, x_and_y):
    regression = False
    if isinstance(model, DecisionTreeRegressor):
        regression = True

    candidates, depths = get_leaf_nodes(model, sort_by_depth=True)

    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    values = model.tree_.value

    feature_names = x.columns
    #used_features_indices = set(model.tree_.feature[model.tree_.feature >= 0])
    #used_features = [feature_names[i] for i in used_features_indices]
    #print(used_features)
    # Identify a leaf with only one sample
    counter = 0
    was_successful = False
    result = model.apply(x)
    for node_index in range(len(candidates)):
        #print("New try")
        # Take the first node with a 1 for simplicity
        target_node = candidates[node_index]
        left = children_left[target_node]
        right = children_right[target_node]
        children_left_values = values[left]
        children_right_values = values[right]
        values_target_node = values[target_node]

        # Find a sample that passes through this node
        leaf_sample_index = None
        for i in range(len(result)):
            # Extract the decision path for this sample
            sample = pd.DataFrame([x.iloc[i].values], columns=x.columns)
            node_indicator = model.decision_path(sample)
            path = node_indicator.indices[
                   node_indicator.indptr[0]:node_indicator.indptr[1]]
            if target_node in path:
                leaf_sample_index = i
                break

        if leaf_sample_index is not None:
            # Extract the decision path for this sample
            sample = pd.DataFrame([x.iloc[leaf_sample_index].values], columns=x.columns)
            node_indicator = model.decision_path(sample)
            path = node_indicator.indices[
                   node_indicator.indptr[0]:node_indicator.indptr[1]]

        # Variables for storing conditions
        feature = model.tree_.feature
        threshold = model.tree_.threshold
        conditions = []

        # delete all nodes which come after the target node
        target_node_index = np.where(path == target_node)[0][0]

        increment = 0
        # if target node is a leaf node we do not want to add a condition
        if path[len(path) - 1] == target_node:
            increment = 1
        path = path[:target_node_index + increment]

        # Construct the query string considering both directions
        for node in path:  # Exclude the last node, which is a leaf
            if feature[node] != -2:  # Check if not a leaf node
                feature_name = x.columns[feature[node]]
                # Determine direction based on whether the left child matches the next node in the path
                if node + 1 in path and children_left[node] == node + 1:
                    condition = f"`{feature_name}` <= {threshold[node]}"
                else:
                    condition = f"`{feature_name}` > {threshold[node]}"
                conditions.append(condition)


        # Join conditions with logical 'AND'
        query_string = " & ".join(conditions)


        # We use only the class information. But as the decision tree does not have saved the class information we have
        # to infer it from the values array
        if not regression:
            for label in [0, 1]:
                query_string_with_label = query_string
                query_string_with_label += " & " + f"`y` == {label}"
                singled_out_sample = x_and_y.query(query_string_with_label)
                counter += 1
                if len(singled_out_sample) == 1:
                    was_successful = True
                    print(query_string_with_label)
                    #print("Singled-Out-Sample:")
                    #print(singled_out_sample)
                    break
        else:
            for sign in ["<=", ">"]:
                query_string_with_label = query_string
                query_string_with_label += " & " + f"`y` {sign} {values[node_index][0][0]}"
                singled_out_sample = x_and_y.query(query_string_with_label)
                counter += 1
                if len(singled_out_sample) == 1:
                    was_successful = True
                    print(query_string_with_label)
                    #print("Singled-Out-Sample:")
                    #print(singled_out_sample)
                    break
        if was_successful:
            break

    if was_successful:
        so_acc = 1 / counter
    else:
        so_acc = 0
    print(f"Singling-Out-ACC: {so_acc}")
    return so_acc

def get_leaf_nodes(tree, sort_by_depth=False):
    # Get the tree structure
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    node_depth = np.zeros(shape=tree.tree_.node_count, dtype=np.int64)
    is_leaves = np.zeros(shape=tree.tree_.node_count, dtype=bool)

    # Initialize stack for depth-first traversal
    stack = [(0, 0)]  # (node_id, depth)

    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If we have a test node
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    # Get the leaf nodes and their depths
    leaf_nodes = np.where(is_leaves)[0]
    leaf_depths = node_depth[leaf_nodes]

    if sort_by_depth:
        # Sort leaf nodes by depth, largest depth first
        leaf_nodes = leaf_nodes[np.argsort(-leaf_depths)]
        leaf_depths = leaf_depths[np.argsort(-leaf_depths)]

    return leaf_nodes, leaf_depths

def evaluate_so_acc_(x_train, y_train, x_test, y_test, df, use_meta_data=True,
                     num_iterations=10, seed=2024):
    # set numpy seed
    np.random.seed(seed)
    seeds = np.random.choice(10000000, num_iterations)

    result_so = 0
    for i, seed in enumerate(seeds):
        if use_meta_data:
            model, acc = train_decision_tree(seed, x_train, y_train, x_test, y_test,
                                             plot_name=f"tree_{i}.png")
            singled_out_sample, query_string = get_so_query_values(model, x_train, df)
            if query_string:
                result_so += 1
        else:
            model, acc = train_decision_tree(seed, x_train, y_train, x_test, y_test)
            result_so += perform_so_attack_without_meta_data(model, x_train, df)
    result_so /= num_iterations

    print("SO-ACC: " + str(result_so))

def evaluate_so_acc_new(num_iterations, path_data, path_models, use_meta_data):
    result_so = 0
    for i in range(num_iterations):
        print(f"Tree {i}")
        train_data = pd.read_csv(path_data + f"{i}_train.csv")
        train_without_label = train_data.drop(columns=['y'])
        model = pickle.load(open(path_models + f"{i}_tree.pkl", 'rb'))

        plot_tree(model, train_without_label, f"tree_{i}.png")

        if use_meta_data:
            singled_out_sample, query_string = get_so_query_values(model, train_without_label, train_data)
            if query_string:
                result_so += 1
        else:
            result_so += perform_so_attack_without_meta_data(model, train_without_label, train_data)
    result_so /= num_iterations

    print("SO-ACC: " + str(result_so))
    return result_so


def perform_experiments(path_data, path_models, num_iterations=100):
    names = ["cardio", "adult", "house"]
    results = []

    for name in names:
        path_data_here = f"{path_data}{name}/"
        path_models_here = f"{path_models}{name}_trees/"
        result_meta_data = evaluate_so_acc_new(num_iterations, path_data_here, path_models_here, True)
        result_not_meta_data = evaluate_so_acc_new(num_iterations, path_data_here, path_models_here, False)
        results.append((result_meta_data, result_not_meta_data))

    for i, name in enumerate(names):
        print(f"Results for dataset {name}")
        print(f"Meta-Data: {results[i][0]}")
        print(f"Not Meta-Data: {results[i][1]}")

def obfuscate_subtree(inner_tree, index):
    """ Recursively obfuscate the entire subtree rooted at the given index. """
    # Set the value of the current node to None or NaN
    inner_tree.value[index] = np.ones(inner_tree.value[index].shape) * -1
    inner_tree.threshold[index] = -1

    # If this node has children, obfuscate them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        obfuscate_subtree(inner_tree, inner_tree.children_left[index])
    if inner_tree.children_right[index] != TREE_LEAF:
        obfuscate_subtree(inner_tree, inner_tree.children_right[index])

def prune_index_helper(inner_tree, index):
    left = inner_tree.children_left[index]
    right = inner_tree.children_right[index]
    if 1 in inner_tree.value[left] or 1 in inner_tree.value[right]:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        #left_values = inner_tree.value[left]
        #right_values = inner_tree.value[right]
        obfuscate_subtree(inner_tree, left)
        obfuscate_subtree(inner_tree, right)
        # if there are shildren, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index_helper(inner_tree, left)
        prune_index_helper(inner_tree, right)

def prune_tree(tree):
    # Deep copy the tree to avoid modifying the original one
    pruned_tree = copy.deepcopy(tree)
    prune_index_helper(pruned_tree.tree_, 0)  # Start from the root node (index 0)
    return pruned_tree

if __name__ == "__main__":
    config_data = read_config(path_to_config)
    path_data = config_data.output_path + "/decision_tree_attack_data/"
    target_path = config_data.output_path + "/decision_tree_attack_trees/"
    perform_experiments(path_data, target_path, num_iterations=100)