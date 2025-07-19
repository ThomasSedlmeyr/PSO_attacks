import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score

from src.decision_tree_attack.attack_decision_tree import prune_tree
from src.utils.utils import read_config, path_to_config


def evaluate_pruned_tree(path_trees, path_dataset, num_trees=100):
    test_df = pd.read_csv(path_dataset)
    x = test_df.drop(columns=['y'])
    y = test_df['y']

    original_scores = np.empty(num_trees)
    pruned_scores = np.empty(num_trees)
    for i in range(num_trees):
        # load pickle file
        with open(path_trees + f"{i}_tree.pkl", 'rb') as file:
            tree = pickle.load(file)

        pruned_tree = prune_tree(tree)
        original_scores[i] = evaluate_tree(x, y, tree)
        pruned_scores[i] = evaluate_tree(x, y, pruned_tree)

    differences = np.abs(original_scores - pruned_scores)
    difference = np.mean(differences)
    std = np.std(differences)
    # print differences
    print(f"Original scores: {np.mean(original_scores)}")
    print(f"Pruned scores: {np.mean(pruned_scores)}")
    print(f"Mean difference: {difference}")
    print(f"Standard deviation: {std}")


def evaluate_tree(x, y, tree, regression=False):
    y_pred = tree.predict(x)
    if regression:
        metric = mean_absolute_error(y, y_pred)
    else:
        metric = accuracy_score(y, y_pred)
    return metric


def evaluate_trees(path_trees, path_datasets, num_trees=100):
    names = ["cardio", "adult"]
    for name in names:
        print(f"Evaluating trees on dataset {name}")
        evaluate_pruned_tree(path_trees + name + "_trees/", path_datasets + name + "_test.csv", num_trees)


if __name__ == "__main__":
    config_data = read_config(path_to_config)
    path_datasets = config_data.output_path + "/decision_tree_attack_data/"
    path_trees = config_data.output_path + "/decision_tree_attack_trees/"
    evaluate_trees(path_trees, path_datasets, num_trees=100)