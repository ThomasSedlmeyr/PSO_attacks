import copy
from typing import Union

import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

from src.analysis.attack_utils import train_attack_model
from src.analysis.querybased import get_queries, extract_querybased

"""
Logic for training and evaluating the attack model. 
"""

class AttackModel:
    def __init__(self, model: Union[RandomForestClassifier, ExtraTreesClassifier, XGBClassifier], threshold, accuracy,
                 auc, partial_auc, target_record, num_input_features):
        self.model = model
        self.threshold = threshold
        self.accuracy = accuracy
        self.auc = auc
        self.partial_auc = partial_auc
        self.target_record = target_record
        self.num_input_features = num_input_features
        self.id = None

    @classmethod
    def build_and_train_model(cls, attack_type, num_shadow_pairs_samples, path_saved_data, path_shadow_x,
                              target_path, num_train_pairs, model_index=0, classifier_type="gradient_boosting"):
        """
        @param num_shadow_pairs_samples: Number of pairs contained in the shadow dataset. One dataset of the pair
        contains the target record while the other does not.
        @param path_y_shadow_datasets: The path to the original sampled datasets
        @param path_x_shadow_datasets: The path to the synthetic datasets
        @param target_path: The path to save the model
        @param num_train_pairs: Number of pairs used for training the classifier
        @param model_index: The index of the model
        @return:
        """
        path_roc_curve = f"{target_path}{model_index}.png"
        clf, best_threshold, acc_test, auc_test, partial_auc, target_record, num_input_features = train_attack_model(
                                                                                    attack_type,
                                                                                    num_shadow_pairs_samples,
                                                                                    path_saved_data,
                                                                                    path_shadow_x,
                                                                                    target_path,
                                                                                    num_train_pairs,
                                                                                    index=model_index,
                                                                                    path_roc_curve=path_roc_curve,
                                                                                    classifier_type=classifier_type)
        model = cls(clf, best_threshold, acc_test, auc_test, partial_auc, target_record, num_input_features)
        model.save(f"{target_path}{model_index}.joblib")
        return model

    def perform_mia(self, input_data, attack_type="black_box", input_features_argument=None):
        """

        @param input_data: For blackbox attacks, this is the synthetic dataset. For whitebox attacks, this is the model
        we would like to attack
        @param attack_type:
        @return:
        """
        if input_features_argument is not None:
            input_features = copy.deepcopy(input_features_argument)
        elif attack_type == "black_box" and input_features_argument is None:
            df_cols = list(input_data.columns)
            # extract features
            queries = get_queries(df_cols)
            input_features = extract_querybased(self.target_record, input_data, queries)
        elif attack_type == "white_box" and input_features_argument is None:
            input_features = input_data.get_wb_features()
            if len(input_features) > self.num_input_features:
                input_features = input_features[:self.num_input_features]
            elif len(input_features) < self.num_input_features:
                input_features += [0] * (self.num_input_features - len(input_features))
        elif not (attack_type == "black_box" or attack_type == "white_box"):
            raise ValueError(f"Invalid attack type: {attack_type}")

        prediction = self.model.predict_proba([input_features])[:, 1]
        binary_score = (prediction >= self.threshold).astype(int)

        return {"prediction": prediction, "binary_score": binary_score, "input_features": input_features}

    def __lt__(self, other):
        return self.auc < other.auc

    def __repr__(self):
        return (f"ModelResult(model={self.model}, threshold={self.threshold}, "
                f"accuracy={self.accuracy}, auc={self.auc}, target_record={self.target_record})")

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            joblib.dump(self, file)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as file:
            return joblib.load(file)
