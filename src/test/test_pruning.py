import pickle
import unittest

import pandas as pd

from src.utils.utils import read_config, path_to_config
from src.decision_tree_attack.attack_decision_tree import extract_vulnerable_nodes, prune_tree
from src.decision_tree_attack.utils import plot_tree


class PruningTest(unittest.TestCase):
    def test_pruning(self):
        print("Test")
        config = read_config(path_to_config)
        path_models = f"{config.path_decision_tree_attack_trees}/adult_trees/"
        path_dataset = f"{config.path_decision_tree_attack_data}/adult/"
        num_iter = 10
        for i in range(num_iter):
            print(f"Tree {i}")
            with open(path_models + f"{i}_tree.pkl", 'rb') as file:
                model = pickle.load(file)

            data = pd.read_csv(f"{path_dataset}{i}_train.csv")
            #plot_tree(model, data, f"tree_{i}_original.png")
            children_left, children_right, vulnerable_nodes_original, values = extract_vulnerable_nodes(model)
            pruned_tree = prune_tree(model)
            children_left, children_right, vulnerable_nodes_pruned, values = extract_vulnerable_nodes(pruned_tree)
            print(f"Original tree: {len(vulnerable_nodes_original)} vulnerable nodes")
            print(f"Pruned tree: {len(vulnerable_nodes_pruned)} vulnerable nodes")

            #plot_tree(pruned_tree, data, f"tree_{i}_pruned.png")
            self.assertEqual(0, len(vulnerable_nodes_pruned))  # add assertion here


if __name__ == '__main__':
    unittest.main()
