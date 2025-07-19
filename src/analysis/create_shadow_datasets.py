import concurrent.futures
import copy
import os
import random
import warnings
from pathlib import Path

import dill
import joblib
import numpy as np
import pandas as pd
from synthesis.synthesizers.marginal import MarginalSynthesizer
from tqdm import tqdm

from src.synthetic_models.dpart_pb import DPartPB
from src.synthetic_models.dpart_synthpop import DPartSynthpop, get_bounds
from src.synthetic_models.dsynth_pb import DSynthPB
from src.synthetic_models.model_container import ModelContainer
from src.synthetic_models.synth_data_gen_pb import SynthDataGenPB
from src.synthetic_models.synthetic_base import SyntheticModel
from src.utils.utils import conv_to_cat, get_metadata

warnings.filterwarnings("ignore", category=UserWarning)


"""
Prepare target synthetic data to be attacked and shadow synthetic data for attacks to be trained on
"""

def build_shadow_pair(shadow_size, df_input, target_idx=0, random_state=42, sample_with_replacement=False):
    """
    Samples a shadow pair from df_input. A shadow pairs consist of two datasets of same length sampled from df_input.
    One dataset contains the target record, while the other does not. The target record is the record which is specified
    by target_idx. The idea is that the caller of this method has all ready sorted the df_input by vulnerability of the
    records. By calling this method multiple times with the same target_idx but different random_state, the caller can
    build a shadow dataset for a specific target record. Containing Multiple shadow pairs, where in half of samples the
    target is contained while in the other half it is not.

    @param shadow_size: The number of samples contained in each of the two datasets of the shadow pair
    @param df_input: The dataset sorted by vulnerability
    @param target_idx: The index of the target record
    @param random_state:
    @param sample_with_replacement:
    @return:
    """

    if isinstance(df_input, str):
        df = pd.read_csv(df_input)
    else:
        df = copy.deepcopy(df_input)

    target_record = df.iloc[[target_idx]]
    #print(target_record)
    df_in = df.sample(n=shadow_size - 1, random_state=random_state, replace=sample_with_replacement).append(
        target_record, ignore_index=True)
    #print(df_in.head(5))
    df_dropped_target = df.drop(target_idx)
    df_out = df_dropped_target.sample(n=shadow_size, random_state=random_state, replace=sample_with_replacement)

    metadata = get_metadata(df)
    
    data = {'df_in': df_in, 'df_out': df_out, 'metadata': metadata}
    return data


def build_shadow_dataset_for_single_target_record(num_shadow_data_pairs, shadow_size, input_path_csv, target_path,
                                                  target_idx=0, start_random_state=0, show_progress=True,
                                                  sample_with_replacement=False, max_workers=22):
    #Path(target_path).mkdir(parents=True, exist_ok=True)
    indices = range(num_shadow_data_pairs)
    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(build_shadow_pair, shadow_size=shadow_size, df_input=input_path_csv,
                                   target_idx=target_idx, random_state=start_random_state + index,
                                   sample_with_replacement=sample_with_replacement) for index in indices]
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_shadow_data_pairs,
                           desc="Sampling Shadow Datasets", disable=not show_progress):
            try:
                # Retrieve the result to catch any exception that occurred
                current_result = future.result()
                all_results.append(current_result)
            except Exception as e:
                # Handle or log the exception here
                print(f"Exception in: build_shadow_dataset_for_single_target_record {e}")
        executor.shutdown(wait=True)
        with open(target_path + "x_in_out_meta_dataset.joblib", 'wb') as file:
            joblib.dump(all_results, file)
        # print("built shadow_dataset")


def build_first_n_shadow_datasets(n, num_shadow_data_pairs, shadow_size, data_set, target_path, sample_with_replacement,
                                  max_workers):
    """
    Builds n shadow datasets containing num_shadow_data_pairs shadow pairs each. A shadow pair consists of two datasets
    where one contains the target record and the other does not. The idea is to build shadow datasets for the first n
    most vulnearable records in the data_set. Therefore, data_set has to be sorted by vulnerability in descending order.

    @param n: The number of shadow datasets to be built
    @param num_shadow_data_pairs: The number of shadow pairs contained in each shadow dataset
    @param shadow_size: The number of samples contained in each of the two datasets of the shadow pair
    @param data_set: The dataset sorted by vulnerability
    @param target_path:
    @param sample_with_replacement:
    @param max_workers:
    @return:
    """

    Path(target_path).mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(n), "Sampling Shadow Datasets"):
        current_path = target_path + str(i) + "_"
        build_shadow_dataset_for_single_target_record(num_shadow_data_pairs, shadow_size, data_set, current_path, i,
                                                      start_random_state=i * num_shadow_data_pairs,
                                                      show_progress=False,
                                                      sample_with_replacement=sample_with_replacement,
                                                      max_workers=max_workers)


def apply_m_to_shadow_pair(num_syn_samples, data, epsilon=1000, model_name="marginal", temp_path="", index=0,
                           n_synth_datasets=1):
    df_in = data["df_in"]
    df_out = data["df_out"]
    #meta_data = data["metadata"]
    meta_data = None
    syn_ins, model_in = apply_m(df_in, meta_data, None, num_syn_samples, epsilon,
                       model_name=model_name, num_synth_datasets=n_synth_datasets)
    syn_outs, model_out = apply_m(df_out, meta_data, None, num_syn_samples, epsilon,
                       model_name=model_name, num_synth_datasets=n_synth_datasets)

    # We have to save the model to the filesystem as it is not possible to return it, because of lambda functions
    # generated in when model.fit is called! After saving and loading the model this error is not present anymore
    model_in.save(f"{temp_path}{index}_model_in")
    model_out.save(f"{temp_path}{index}_model_out")

    result_tuple_list = []
    for i in range(n_synth_datasets):
        tuple = (syn_ins[i], syn_outs[i])
        result_tuple_list.append(tuple)

    return result_tuple_list


def apply_m(data, meta_data=None, output_path=None, num_syn_samples=None, epsilon=10, model_name="marginal", seed=42,
            categorical_data_only=True, num_synth_datasets=1):

    if num_syn_samples is None:
        num_syn_samples = len(data)
    if categorical_data_only:
        data = conv_to_cat(data)

    np.random.seed(seed)
    random.seed(seed)

    if model_name == "privbayes_dpart":
        bounds = get_bounds(data)
        if meta_data is None:
            meta_data = get_metadata(data)
        model = DPartPB(epsilon=epsilon, metadata=meta_data, n_parents=2, bounds=bounds)
    elif model_name == "privbayes_syn_gen":
        model = SynthDataGenPB(epsilon=epsilon, verbose=False)
    elif model_name == "privbayes_data_syn":
        if meta_data is None:
            meta_data = get_metadata(data)
        model = DSynthPB(epsilon=epsilon, metadata=meta_data, n_parents=2, seed=seed)
    elif model_name == "synthpop":
        bounds = get_bounds(data)
        model = DPartSynthpop(epsilon=epsilon, bounds=bounds)
    elif model_name == "marginal":
        model = MarginalSynthesizer(epsilon=epsilon, verbose=False)
    else:
        raise ValueError("Unknown model_name")

    model.fit(data)

    synth_dfs = []
    for i in range(num_synth_datasets):
        if model_name == "privbayes_data_syn":
            if i < num_synth_datasets - 1:
                sampled_df = model.sample(num_syn_samples, remove_desc=False)
            else:
                sampled_df = model.sample(num_syn_samples, remove_desc=False)
        else:
            sampled_df = model.sample(num_syn_samples)

        if categorical_data_only:
            sampled_df = conv_to_cat(sampled_df)
        sampled_df.columns = sampled_df.columns.astype(str)
        synth_dfs.append(sampled_df)


    if output_path is not None:
        synth_dfs[0].to_csv(output_path, index=False)

    return synth_dfs, model


def apply_m_in_parallel_to_single_shadow_dataset(num_shadow_pairs, num_syn_samples, input_path, target_path,
                                                 target_path_models, epsilon, show_progress=True, model_name="marginal",
                                                 max_workers=22, temp_path="", n_synthetic_datasets=1):
    indices = range(num_shadow_pairs)
    with open(input_path, 'rb') as file:
        sampled_data = joblib.load(file)

    result_list = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(apply_m_to_shadow_pair, num_syn_samples, sampled_data[index],
                                   epsilon, model_name, temp_path=temp_path, index=index,
                                   n_synth_datasets=n_synthetic_datasets) for index in indices]
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_shadow_pairs, desc="Building Shadow Datasets", disable=not show_progress):
            try:
            # Retrieve the result to catch any exception that occurred
                result = future.result()
                result_list.extend(result)
            except Exception as e:
                # Handle or log the exception here
                print(f"Exception in: apply_m_in_parallel_to_single_shadow_dataset {e}")
        executor.shutdown(wait=True)

    with open(target_path + "y_in_out_dataset.joblib", 'wb') as file:
        joblib.dump(result_list, file)


    models_in, models_out = [], []
    for index in indices:
        model_in = SyntheticModel.load(f"{temp_path}{index}_model_in")
        models_in.append(model_in)
        model_out = SyntheticModel.load(f"{temp_path}{index}_model_out")
        models_out.append(model_out)

    container = ModelContainer(models_in, models_out)
    container.save(target_path_models)

    for index in indices:
        os.remove(f"{temp_path}{index}_model_in")
        os.remove(f"{temp_path}{index}_model_out")


def apply_m_to_single_shadow_dataset_sequential(num_shadow_pairs, num_syn_samples, input_path, target_path,
                                                 target_path_models, epsilon, show_progress=True, model_name="marginal",
                                                 temp_path="", n_synthetic_datasets=1):
    indices = range(num_shadow_pairs)
    with open(input_path, 'rb') as file:
        sampled_data = joblib.load(file)

    datasets = []
    for index in tqdm(indices, desc="Building Shadow Datasets", disable=not show_progress):
        result = apply_m_to_shadow_pair(num_syn_samples, sampled_data[index], epsilon, model_name,
                                        temp_path=temp_path, index=index, num_synth_datasets=n_synthetic_datasets)
        datasets.extend(result)

    models_in, models_out = [], []
    for index in indices:
        models_in.append(DPartPB.load(f"{temp_path}{index}_model_in"))
        models_out.append(DPartPB.load(f"{temp_path}{index}_model_out"))

    models_dict = {"models_in": models_in, "models_out": models_out}
    dill.dump(models_dict, open(target_path_models + "models_in_and_out.dill", 'wb'))

    for index in indices:
        os.remove(f"{temp_path}{index}_model_in")
        os.remove(f"{temp_path}{index}_model_out")

    with open(target_path + "y_in_out_dataset.joblib", 'wb') as file:
        joblib.dump(datasets, file)


def apply_m_to_multiple_shadow_datasets(num_shadow_datasets, num_shadow_pairs, num_syn_samples, input_path, target_path,
                                        target_path_models, epsilon, model_name="marginal", max_workers=22, temp_path="",
                                        parallel=True, n_synthetic_datasets=1):
    Path(target_path).mkdir(parents=True, exist_ok=True)
    Path(target_path_models).mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(num_shadow_datasets), "Applying M to Shadow Datasets"):
        input_file_path = f"{input_path}{i}_x_in_out_meta_dataset.joblib"
        target_file_path = f"{target_path}{i}_"
        target_file_path_models = f"{target_path_models}{i}_"
        if parallel:
            apply_m_in_parallel_to_single_shadow_dataset(num_shadow_pairs, num_syn_samples, input_file_path,
                                                         target_file_path, target_file_path_models, show_progress=False,
                                                         epsilon=epsilon, model_name=model_name, max_workers=max_workers,
                                                         temp_path=temp_path, n_synthetic_datasets=n_synthetic_datasets)
        else:
            apply_m_to_single_shadow_dataset_sequential(num_shadow_pairs, num_syn_samples, input_file_path,
                                                        target_file_path, target_path_models, epsilon,
                                                        show_progress=True, model_name=model_name, temp_path=temp_path,
                                                        n_synthetic_datasets=n_synthetic_datasets)

if __name__ == '__main__':
    shadow_size = 1000
    num_syn_samples = 1000
    num_shadow_pairs = 100
    num_shadow_datasets = 5
    epsilon = 1000
    input_path_csv = "/datasets/adult/adult_cat.csv"
    input_path_vuln = "/datasets/adult/vulns.txt"
    target_path_sampled_shadow_datasets = "/mnt/data/Dokumente/Studium/Master/Semester_4/Guided_Research/own_code/PSO_attacks/output/first_10_shadow_datasets/samples/"
    target_path_m_datasets = "/mnt/data/Dokumente/Studium/Master/Semester_4/Guided_Research/own_code/PSO_attacks/output/first_10_shadow_datasets/m/"

    #target_path = "/mnt/data/Dokumente/Studium/Master/Semester_4/Guided_Research/own_code/PSO_attacks/output/in_m/"
    #target_out_path = "/mnt/data/Dokumente/Studium/Master/Semester_4/Guided_Research/own_code/PSO_attacks/output/out_m/"

    #build_shadow_pair(shadow_size, input_path_csv, input_path_vuln, target_path, target_idx=0, use_vul_target=True, random_state=42)
    #apply_m_to_shadow_pair(num_syn_samples, target_path, target_in_path, target_out_path, 0)
    #build_shadow_dataset_for_single_target_record(num_shadow_datasets, shadow_size, input_path_csv, input_path_vuln, target_path, target_idx=0, use_vul_target=True)
    #build_first_n_shadow_datasets(num_shadow_datasets, num_shadow_pairs, shadow_size, input_path_csv, input_path_vuln, target_path_sampled_shadow_datasets, use_vul_target=True)
    #apply_m_in_parallel(num_shadow_datasets, num_syn_samples, target_path, target_path)
    apply_m_to_multiple_shadow_datasets(num_shadow_datasets, num_shadow_pairs, num_syn_samples, target_path_sampled_shadow_datasets,
                                        target_path_m_datasets, epsilon)