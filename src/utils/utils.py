from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
import uuid


path_to_config = "../config.yaml"

def read_config(file_path=None):
    if file_path is None:
        file_path = path_to_config
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    config_dict["n_synth"] = config_dict["size_x"]
    config_dict["path_csv"] = config_dict["path_data"] + config_dict["dataset_name"] + "/" + config_dict["dataset_name"] + "_cat.csv"
    #config_dict["path_vuln"] = config_dict["path_data"] + config_dict["dataset_name"] + "/" + "vulns.txt"
    #config_dict["output_path"] += config_dict["experiment_name"] + "/"
    config_dict["path_d"] = config_dict["output_path"] + "d.csv"
    config_dict["path_shadow_x"] = config_dict["output_path"] + "shadow_x" + "/"
    config_dict["path_shadow_y"] = config_dict["output_path"] + "shadow_y" + "/"
    config_dict["path_models"] = config_dict["output_path"] + "models" + "/"
    config_dict["path_sampled_data"] = config_dict["output_path"]
    config_dict["path_attack_models"] = config_dict["output_path"] + "attack_models/"
    config_dict["path_black_box_attack_models"] = config_dict["path_attack_models"] + "black_box/"
    config_dict["path_white_box_attack_models"] = config_dict["path_attack_models"] + "white_box/"
    config_dict["temp_path"] = config_dict["temp_path"] + config_dict["experiment_name"] +f"_{uuid.uuid4()}/"
    Path(config_dict["temp_path"]).mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(**config_dict)


def adjust_output_path_dependencies(config):
    config.path_shadow_y = config.output_path + "shadow_y/"
    config.path_attack_models = config.output_path + "attack_models/"
    config.path_models = config.output_path + "models/"
    config.path_black_box_attack_models = config.path_attack_models + "black_box/"
    config.path_white_box_attack_models = config.path_attack_models + "white_box/"

def conv_to_cat(df):
    # convert all attributes of dataset to categorical attributes
    new_df = df.copy()
    for col in df.columns:
        new_df[col] = new_df[col].astype(str).astype('category')
    return new_df


def get_metadata(df):
    # extract metadata from dataset (assume all attributes are categorical)
    df = conv_to_cat(df)
    return {
        'columns': [
            {
                'name': col,
                'type': 'Categorical',
                'i2s': list(df[col].unique())
            }
            for col in df.columns
        ]
    }
