from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Any

from tqdm import tqdm

from src.analysis.attack_model import AttackModel

"""
Contains the logic to train, perform inference on multiple attack models
"""

class AttackModelContainer:
    def __init__(self, attack_models: List[AttackModel]):
        self.attack_models = attack_models

    @classmethod
    def build_and_train_attack_models(cls, attack_type, num_attack_models, num_shadow_pairs_samples, path_saved_data,
                                      path_shadow_x, target_path, train_fraction=0.8, max_workers=20,
                                      classifier_model="gradient_boosting"):
        Path(target_path).mkdir(parents=True, exist_ok=True)
        num_train_pairs = int(num_shadow_pairs_samples * train_fraction)
        attack_models = []
        if classifier_model == "gradient_boosting":
            # We have to make this distinction because the gradient boosting classifier can not be used in a ProcessPoolExecutor
            for i in range(num_attack_models):
                AttackModel.build_and_train_model(attack_type, num_shadow_pairs_samples, path_saved_data,
                                                  path_shadow_x, target_path, num_train_pairs, model_index=i,
                                                  classifier_type=classifier_model)
            return cls(attack_models)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        AttackModel.build_and_train_model,
                        attack_type,
                        num_shadow_pairs_samples,
                        path_saved_data,
                        path_shadow_x,
                        target_path,
                        num_train_pairs,
                        model_index=i,
                        classifier_type=classifier_model
                    ) for i in range(num_attack_models)
                ]

                for future in tqdm(as_completed(futures), total=num_attack_models, desc="Training Attack Models"):
                    try:
                        # Retrieve the result to catch any exception that occurred
                        model = future.result()
                        attack_models.append(model)
                    except Exception as e:
                        # Handle or log the exception here
                        print(f"Exception in: build_and_train_attack_models {e}")
            return cls(attack_models)

    def __repr__(self):
        result = ""
        for i, attack_model in enumerate(self.attack_models):
            result += "Model: " + str(i) + "\n" + repr(attack_model) + "\n"

        return result

    @staticmethod
    def load(folder_path):
        paths_attack_models = Path(folder_path).rglob("*.joblib")
        attack_models = []
        for path_model in paths_attack_models:
            attack_models.append(AttackModel.load(path_model))
        attack_models = sorted(attack_models, reverse=True)

        for i in range(len(attack_models)):
            attack_models[i].id = i

        return AttackModelContainer(attack_models)

    def save(self, folder_path):
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        for i, attack_model in enumerate(self.attack_models):
            attack_model.save(folder_path + str(i) + ".joblib")

    def identify_best_target_record_and_model(self, attack_type, dataset_y, weight_score_by_metric=False, input_features=None) -> \
    Tuple[Any, AttackModel, List[Any]]:
        """
        Identifies the best target record with its model for a given dataset_y

        @param dataset_y:
        @param weight_score_by_metric:
        @param input_features: Optional precomputed input features for saving compute time
        @return:
        """
        models_predictions_and_scores = []
        new_input_features = []
        for i, attack_model in enumerate(self.attack_models):
            if input_features is not None:
                mia_result = attack_model.perform_mia(dataset_y, attack_type, input_features[i])
            else:
                mia_result = attack_model.perform_mia(dataset_y, attack_type, None)
            if weight_score_by_metric:
                models_predictions_and_scores.append((mia_result["prediction"], attack_model,
                                                      mia_result["prediction"] * attack_model.auc))
            else:
                models_predictions_and_scores.append((mia_result["prediction"], attack_model, mia_result["prediction"]))
            new_input_features.append(mia_result["input_features"])

        best_model_prediction_and_score = max(models_predictions_and_scores, key=lambda x: x[2])
        return best_model_prediction_and_score[0], best_model_prediction_and_score[1], new_input_features
