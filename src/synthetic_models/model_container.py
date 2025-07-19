from typing import List

import dill

from src.synthetic_models.synthetic_base import SyntheticModel


def load(path):
    result_dict = dill.load(open(path, 'rb'))

    for model in result_dict["models_in"]:
        model.after_load()
    for model in result_dict["models_out"]:
        model.after_load()

    return result_dict


class ModelContainer:

    def __init__(self, in_models: List[SyntheticModel], out_models: List[SyntheticModel]):
        self.in_models = in_models
        self.out_models = out_models

    def save(self, path):
        for model in self.in_models:
            model.before_save()
        for model in self.out_models:
            model.before_save()

        models_dict = {"models_in": self.in_models, "models_out": self.out_models}
        dill.dump(models_dict, open(path + "models_in_and_out.dill", 'wb'))
