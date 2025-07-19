import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
#from diffprivlib.models import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from src.decision_tree_attack.parse_data import read_sampled_dataset
from src.utils.utils import read_config, path_to_config


def train_model(x_train, y_train, seed=42, plot_name="tree.png"):
    clf = DecisionTreeClassifier(random_state=seed, min_samples_leaf=1, max_depth=10)
    #clf = DecisionTreeClassifier(random_state=42, min_samples_split=5, min_samples_leaf=5, max_depth=5)

    #clf = RandomForestClassifier(n_estimators=3)

    # Train the model
    clf.fit(x_train, y_train)
    #clf = clf.estimators_[0]
    #clf.feature_names_in_ = x_train.columns

    tree.plot_tree(clf, filled=True, feature_names=x_train.columns, class_names=['0', '1'])
    plt.savefig(plot_name, dpi=700)
    #plt.show(dpi=600)

    y_pred = clf.predict(x_train)
    accuracy = np.mean(y_pred == y_train["y"])
    print(accuracy)
    return clf


def train_decision_tree(id, x_train, y_train, x_test, y_test, use_cross_val=True, plot_name="tree.png",
                        target_path=None, n_jobs=20, tree_type="decision_tree"):
    if use_cross_val:
        param_dist = {
            # 'max_depth': np.arange(1, 64),
            # 'min_samples_split': np.arange(2, 20),
            # 'min_samples_leaf': np.arange(1, 10),
            'max_depth': np.arange(5, 64),
            'min_samples_split': np.arange(2, 32),
            'min_samples_leaf': np.arange(1, 16),
            #'max_depth': np.arange(2, 64),
            #'min_samples_split': np.arange(2, 4),
            #'min_samples_leaf': np.arange(1, 1),
        }
        # Initialize the DecisionTreeClassifier
        if tree_type == "decision_tree":
            dt = DecisionTreeClassifier(random_state=id)
        elif tree_type == "regression_tree":
            dt = DecisionTreeRegressor(random_state=id)
        else:
            raise ValueError("tree_type must be either 'decision_tree' or 'regression_tree'")
        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            dt, param_distributions=param_dist, n_iter=100, cv=4, verbose=1, random_state=id, n_jobs=n_jobs
        )    # Fit RandomizedSearchCV
        random_search.fit(x_train, y_train)
        # Get the best estimator
        best_dt = random_search.best_estimator_
    else:
        best_dt = train_model(x_train, y_train, id, plot_name=plot_name)

    y_pred = best_dt.predict(x_test)
    if tree_type == "regression_tree":
        metric = mean_absolute_error(y_test, y_pred)
    elif tree_type == "decision_tree":
        metric = accuracy_score(y_test, y_pred)

    if target_path:
        pickle_file = target_path + f"{id}_tree.pkl"
        with open(pickle_file, 'wb') as file:
            pickle.dump(best_dt, file)
        # save best hyper parameters and validation accuracy to txt file
        with open(target_path + f"{id}_hyperparams.txt", 'w') as file:
            file.write(f"Best hyperparameters: {best_dt.get_params()}\n")
            file.write(f"Validation metric value: {metric}\n")

    return best_dt, metric

def fit_trees_on_one_dataset(num_trees, path_datasets, target_path, tree_type="decision_tree"):
    Path(target_path).mkdir(parents=True, exist_ok=True)

    for i in range(num_trees):
        x_train, y_train, x_val, y_val = read_sampled_dataset(path_datasets, i, split_in_y_and_x=True)
        train_decision_tree(i, x_train, y_train, x_val, y_val, use_cross_val=True, plot_name=None,
                            target_path=target_path, n_jobs=20, tree_type=tree_type)

def fit_trees_on_all_datasets(num_trees, path_datasets, target_path):
    names_datasets = ["house","cardio", "adult"]

    for name in names_datasets:
        print(f"Fitting trees on dataset {name}")
        if name == "house":
            tree_type = "regression_tree"
        else:
            tree_type = "decision_tree"
        fit_trees_on_one_dataset(num_trees, path_datasets + name + "/", target_path + name + "_trees/", tree_type)

def evaluate_majority_predictor(y_train, y_val):
    # Identify the majority class in the training set
    majority_class = y_train["y"].mode()[0]

    # Create predictions where all predictions are the majority class
    y_pred = [majority_class] * len(y_val)

    # Calculate the accuracy of the majority class predictor
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Majority class predictor accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    config_data = read_config(path_to_config)
    path_datasets = config_data.output_path + "/decision_tree_attack_data/"
    target_path = config_data.output_path + "/decision_tree_attack_trees/"
    fit_trees_on_all_datasets(100, path_datasets, target_path)