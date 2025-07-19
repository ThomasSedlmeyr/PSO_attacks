import argparse
import concurrent.futures
import copy
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import dill
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.analysis.attack_model import AttackModel
from src.analysis.attack_model_container import AttackModelContainer
from src.analysis.create_shadow_datasets import apply_m, apply_m_to_multiple_shadow_datasets, build_first_n_shadow_datasets
from src.data_processing.vulnerability_computation import compute_vulnerability
from src.utils.utils import conv_to_cat, get_metadata, read_config, adjust_output_path_dependencies
from src.synthetic_models.synthetic_base import SyntheticModel


def get_D_and_sort_by_vuln(path_data, num_samples, randoms_state=42, target_path=None, convert_to_cat=True, num_threads=None):
    df = pd.read_csv(path_data)
    # remove duplicates
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    #print(len(df))
    d = df.sample(n=num_samples, random_state=randoms_state, replace=False)

    if convert_to_cat:
        d = conv_to_cat(d)
    d_sorted = d.copy(deep=True)
    vulns = compute_vulnerability(d_sorted, target_path=None, multi_processing=False, number_threads=None, show_progress=True)
    #print("computing vulnerability")
    #vulns = compute_distances_fast(df, num_threads)
    d_sorted['vuln'] = vulns
    d_sorted = d_sorted.sort_values('vuln', ascending=False)
    d_sorted = d_sorted.drop(columns='vuln')
    d_sorted = d_sorted.reset_index(drop=True)

    if target_path:
        d.to_csv(f"{target_path}d.csv", index=False)
        d_sorted.to_csv(f"{target_path}d_sorted.csv", index=False)
    return d, d_sorted

def get_X_M_D_and_meta_data_old(path_data, size_m, size_d, output_path, epsilon, random_state=42, input_path_vuln=None):
    df = pd.read_csv(path_data)
    if input_path_vuln is not None:
        vulns = np.genfromtxt(input_path_vuln)
        df['vuln'] = vulns

    d_sorted_by_vuln = df.drop_duplicates().sort_values('vuln', ascending=False).head(size_d)
    d_sorted_by_vuln = d_sorted_by_vuln.drop(columns='vuln')
    d = d_sorted_by_vuln
    d = conv_to_cat(d)
    x = d.sample(n=size_m, random_state=random_state, replace=True)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    meta_data = get_metadata(d)
    m = apply_m(d, meta_data, output_path + "d.csv", size_m, epsilon, model_name="dpart_pb")
    d.to_csv(output_path + "d.csv", index=False)
    d_sorted_by_vuln.to_csv(output_path + "d_sorted_by_vuln.csv", index=False)
    x.to_csv(output_path + "x.csv", index=False)

    return x, m, d, d_sorted_by_vuln, meta_data


def get_X_Y_D_and_meta_data(path_data, size_x, size_m, size_d, output_path, epsilon, model_name, random_state=42,
                            input_path_vuln=None, read_from_file=False):
    if read_from_file:
        d_sorted_by_vuln = pd.read_csv(
            "/mnt/data/Dokumente/Studium/Master/Semester_4/Guided_Research/own_code/PSO_attacks/output/d_sorted_by_vuln.csv")
        d = pd.read_csv(
            "/mnt/data/Dokumente/Studium/Master/Semester_4/Guided_Research/own_code/PSO_attacks/output/d.csv")
        y = pd.read_csv(
            "/mnt/data/Dokumente/Studium/Master/Semester_4/Guided_Research/own_code/PSO_attacks/output/y.csv")
        x = pd.read_csv(
            "/mnt/data/Dokumente/Studium/Master/Semester_4/Guided_Research/own_code/PSO_attacks/output/x.csv")
        # d_sorted_by_vuln = pd.read_csv("/mnt/data/Dokumente/Studium/Master/Semester_4/Guided_Research/own_code/PSO_attacks/output/example_experiment/d_sorted_by_vuln.csv")
        # d = pd.read_csv("/mnt/data/Dokumente/Studium/Master/Semester_4/Guided_Research/own_code/PSO_attacks/output/example_experiment/d.csv")
        # m = pd.read_csv("/mnt/data/Dokumente/Studium/Master/Semester_4/Guided_Research/own_code/PSO_attacks/output/example_experiment/m.csv")
        # x = pd.read_csv("/mnt/data/Dokumente/Studium/Master/Semester_4/Guided_Research/own_code/PSO_attacks/output/example_experiment/x.csv")
        meta_data = get_metadata(d)
        d_sorted_by_vuln = conv_to_cat(d_sorted_by_vuln)
        d = conv_to_cat(d)
        y = conv_to_cat(y)
        x = conv_to_cat(x)
        return x, y, d, d_sorted_by_vuln, meta_data

    df = pd.read_csv(path_data)
    if input_path_vuln is not None:
        vulns = np.genfromtxt(input_path_vuln)
        df['vuln'] = vulns

    # get dataset d of size size_d
    d = df.sample(n=size_d, random_state=random_state, replace=False)
    # d = df.drop_duplicates().sort_values('vuln', ascending=False)
    # d = df.head(size_d)
    if input_path_vuln is not None:
        d_sorted_by_vuln = d.sort_values('vuln', ascending=False)
        d = d.drop(columns='vuln')
        d_sorted_by_vuln = d_sorted_by_vuln.drop(columns='vuln')
    else:
        d_sorted_by_vuln = copy.deepcopy(d)

    d = conv_to_cat(d)
    d.reset_index(drop=True, inplace=True)
    d_sorted_by_vuln = conv_to_cat(d_sorted_by_vuln)
    d_sorted_by_vuln.reset_index(drop=True, inplace=True)

    x = d.sample(n=size_x, random_state=random_state, replace=False)
    x = conv_to_cat(x)
    x.reset_index(drop=True, inplace=True)
    # x = copy.deepcopy(d)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    meta_data = get_metadata(d)
    y, model = apply_m(x, meta_data, output_path + "y.csv", size_m, epsilon, model_name=model_name)
    del model
    d.to_csv(output_path + "d.csv", index=False)
    d_sorted_by_vuln.to_csv(output_path + "d_sorted_by_vuln.csv", index=False)
    x.to_csv(output_path + "x.csv", index=False)
    return x, y, d, d_sorted_by_vuln, meta_data

def sample_singling_out_datasets(D: pd.DataFrame, num_samples, num_singling_out_datasets, with_replacement=False,
                                target_path=None, seeds=(1234, 4321)):

    names = ["val", "test"]
    for name, seed in zip(names, seeds):
        singling_out_datasets = []
        np.random.seed(seed)
        random_states = np.random.randint(0, 2 ** 32 - 1, size=num_singling_out_datasets)

        for i in range(num_singling_out_datasets):
            singling_out_datasets.append(D.sample(n=num_samples, replace=with_replacement, random_state=random_states[i]))

        if target_path:
            with open(target_path + f"{name}_singling_out_dataset_x.joblib", 'wb') as file:
                joblib.dump(singling_out_datasets, file)




def apply_m_to_singling_out_dataset(x, epsilon, model_name="marginal", model_temp_path=None):
    y_list, model = apply_m(x, None, None, len(x), epsilon, model_name=model_name)
    # We have to save the model to the filesystem as it is not possible to return it, because of lambda functions
    # generated in when model.fit is called! After saving and loading the model this error is not present anymore
    if model_temp_path:
        model.save(model_temp_path)
    return x, y_list[0]


def perform_mias(m, path_attack_models, metadata):
    attack_models = parse_attack_models(path_attack_models)
    full_uniq_vals = {cdict['name']: cdict['i2s'] for cdict in metadata['columns']}
    result_list = []

    for attack_model in attack_models:
        pred_result = attack_model.perform_mia(m, full_uniq_vals, attack_type="querybased")
        result_list.append({"prediction": pred_result["prediction"], "binary_score": pred_result["binary_score"],
                            "attack_model": attack_model})
    return result_list


def simulate_random_model(path_attack_models, metadata, fraction, seed):
    attack_models = parse_attack_models(path_attack_models)
    result_list = []

    random.seed(seed)

    def generate_next(fraction):
        return 1 if random.random() < fraction else 0

    for attack_model in attack_models:
        simulated_prediction = generate_next(fraction)
        result_list.append(
            {"prediction": simulated_prediction, "binary_score": simulated_prediction, "attack_model": attack_model})
    return result_list


def parse_attack_models(path_attack_models) -> List[AttackModel]:
    attack_models = []
    paths_attack_models = Path(path_attack_models).rglob("*.joblib")
    for path_model in paths_attack_models:
        attack_models.append(AttackModel.load(path_model))
    attack_models = sorted(attack_models)
    return attack_models


def evaluate_singling_out_attack(x, m, path_attack_models, metadata):
    result_list = perform_mias(m, path_attack_models, metadata)
    result_membership = check_if_membership_is_correct(x, result_list)
    print(result_membership)


def check_if_membership_is_correct(x, result_list):
    tp, fp, tn, fn = 0, 0, 0, 0
    for result in result_list:
        if check_if_target_record_is_in_x(result["attack_model"].target_record, x):
            if result["binary_score"] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if result["binary_score"] == 1:
                fp += 1
            else:
                tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def visualize_membership(result_dict, result_path):
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame([result_dict])

    # Melt the DataFrame to long format for plotting
    df_melted = df.melt(var_name='Type', value_name='Count')

    # Plotting the histogram using seaborn
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(df_melted, x='Type', weights='Count', multiple="dodge", bins=20, kde=False)

    # Customize the plot
    num_records = sum(result_dict.values())
    plt.title(
        f"Histogram of analyzed {num_records} records where {result_dict['tp'] + result_dict['fn']} records were used to produce M \n Size X=100, Size D=500, Epsilon=10000")
    plt.xlabel("Type")
    plt.ylabel("Count")

    # Annotate the bars with the concrete numbers
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center',
                    xytext=(0, 10),
                    textcoords='offset points')

    # Save the plot with 400 DPI
    plt.tight_layout()
    plt.savefig(result_path + "overview.png", dpi=400)


def check_if_target_record_is_in_x(target_record: pd.DataFrame, x: pd.DataFrame):
    #print("Target Record")
    # print(target_record.columns)
    #print(target_record)
    #print("X")
    # print(x.columns)
    #print(x)
    small_row = target_record.iloc[0]
    is_row_in_large_df = x.apply(lambda row: row.equals(small_row), axis=1).any()
    return is_row_in_large_df


def apply_m_to_singling_out_datasets(path_sampled_data, model_name, epsilon, num_repetitions, output_path, temp_path,
                                     max_workers=22, name="val"):
    """
    Samples or loads the synthetic datasets which are used to compute the final singling-out accuracy
    @param d:
    @param num_samples:
    @param meta_data:
    @param model_name:
    @param epsilon:
    @param num_repetitions:
    @param output_path:
    @param sample_with_replacement:
    @param start_random_state:
    @param max_workers:
    @return:
    """
    path_stored_x_datasets = path_sampled_data + f"{name}_singling_out_dataset_x.joblib"
    path_synthetic_dataset_y = output_path + f"{name}_singling_out_dataset_y.joblib"
    path_stored_models = output_path + f"{name}_singling_out_test_models.dill" # The path where we save the models we use for
    # generating the synthetic dataset used for computing the singling-out accuracy

    if Path(path_synthetic_dataset_y).exists() and Path(path_stored_models).exists():
        with open(path_synthetic_dataset_y, 'rb') as file:
            synthetic_datasets = joblib.load(file)
        with open(path_stored_models, 'rb') as file:
            models = dill.load(file)

        return synthetic_datasets, models

    with open(path_stored_x_datasets, 'rb') as file:
        singling_out_datasets = joblib.load(file)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        synthetic_datasets = []
        futures = [
            executor.submit(apply_m_to_singling_out_dataset, singling_out_datasets[i], epsilon, model_name,
                            f"{temp_path}{i}") for i in range(num_repetitions)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_repetitions,
                           desc="Building Synthetic Datasets"):
            try:
                # Retrieve the result to catch any exception that occurred
                result = future.result()
                synthetic_datasets.append(result)
            except Exception as e:
                # Handle or log the exception here
                print(f"Exception in: apply_m_in_parallel_to_single_shadow_dataset {e}")

    with open(path_synthetic_dataset_y, 'wb') as file:
        joblib.dump(synthetic_datasets, file)

    models = []
    for i in range(num_repetitions):
        models.append(SyntheticModel.load(f"{temp_path}{i}"))
    dill.dump(models, open(path_stored_models, 'wb'))

    for i in range(num_repetitions):
        os.remove(f"{temp_path}{i}")

    return synthetic_datasets, models


def compute_singling_out_acc(path_sampled_data, attack_type, model_name, epsilon, attack_models_path, num_repetitions,
                             output_path, temp_path, max_workers=22, name="val", show_progress=True):
    x_y_list, synthetic_models = apply_m_to_singling_out_datasets(path_sampled_data, model_name, epsilon, num_repetitions, output_path, temp_path,
                                                                  max_workers=max_workers, name=name)
    correct_singled_out = 0
    attack_model_container = AttackModelContainer.load(attack_models_path)
    target_records = [model.target_record for model in attack_model_container.attack_models]
    #count_number_of_target_records_are_contained_in_so_datasets(target_records, x_y_list)

    attack_model_dict = {}
    
    features_list = None
    # feature_list_path = f"input_features_so_{name}.joblib"
    # if Path(output_path + feature_list_path).exists():
    #     features_list = joblib.load(output_path + feature_list_path)
    #     if len(features_list) != num_repetitions:
    #         print("Number of features list does not match the number of repetitions. The feature list will be recomputed.")
    #         features_list = None
    #     elif len(features_list[0]) != len(attack_model_container.attack_models):
    #         print("Number of features in the feature list does not match the number of attack models. The feature list will be recomputed.")
    #         features_list = None

    #features_list_new = []
    input_features = None
    so_result_list = []
    for i in tqdm(range(num_repetitions), desc="Performing singling out attack", disable=not show_progress):
        x, y = x_y_list[i]
        #print(x.head(5))
        model = synthetic_models[i]
        if features_list is not None:
            input_features = features_list[i]
        if attack_type == "black_box":
            score, model, input_features_new = attack_model_container.identify_best_target_record_and_model(attack_type, y, weight_score_by_metric=False,
                                                                                               input_features=input_features)
        elif attack_type == "white_box":
            score, model, input_features_new  = attack_model_container.identify_best_target_record_and_model(attack_type, model, weight_score_by_metric=False,
                                                                                                input_features=input_features)
        else:
            raise ValueError(f"Invalid attack type: {attack_type}")
        
        #features_list_new.append(input_features_new)

        if model.id in attack_model_dict:
            attack_model_dict[model.id] += 1
        else:
            attack_model_dict[model.id] = 1

        correctly_so = check_if_target_record_is_in_x(model.target_record, x)
        if correctly_so:
            correct_singled_out += 1

        so_result_list.append({"score": score, "model": model, "correctly_so": correctly_so})

    #joblib.dump(features_list_new, output_path + feature_list_path)

    print("Number of times each attack model was selected:")
    for key in sorted(attack_model_dict.keys()):
        print(f"Model_{key}: {attack_model_dict[key]}")

    singling_out_acc = correct_singled_out / num_repetitions
    print(f"Singling out accuracy: {singling_out_acc}")
    return singling_out_acc, so_result_list, attack_model_container

    singling_out_acc = correct_singled_out / num_repetitions
    print(f"Singling out accuracy: {singling_out_acc}")
    return singling_out_acc


def perform_attack(x_y, synthetic_model, attack_type, attack_models):
    x, y = x_y
    if attack_type == "black_box":
        score, model = attack_models.identify_best_target_record_and_model(attack_type, y, weight_score_by_metric=False)
    elif attack_type == "white_box":
        score, model = attack_models.identify_best_target_record_and_model(attack_type, synthetic_model, weight_score_by_metric=False)
    else:
        raise ValueError(f"Invalid attack type: {attack_type}")

    return check_if_target_record_is_in_x(model.target_record, x)

def count_number_of_target_records_are_contained_in_so_datasets(target_records, singling_out_datasets):
    for i, target_record in enumerate(target_records):
        num_target_records = 0
        for x, y in singling_out_datasets:
            if check_if_target_record_is_in_x(target_record, x):
                num_target_records += 1
        print(f"Number of target record {i} is contained in singling out dataset: {num_target_records}")

def analyse_result_so_list(result_list: List[Dict], attack_models: List[AttackModel]):
    dict_wrong_counts = {result["model"].id : 0 for result in result_list}
    dict_correct_counts = {result["model"].id : 0 for result in result_list}

    # initialize the dictionaries


    for result in result_list:
        if result["correctly_so"]:
            if result["model"].id in dict_correct_counts:
                dict_correct_counts[result["model"].id] += 1
        else:
            if result["model"].id in dict_wrong_counts:
                dict_wrong_counts[result["model"].id] += 1

    bad_ids = []
    for key in sorted(dict_correct_counts.keys()):
        tpr = dict_correct_counts[key] / (dict_correct_counts[key] + dict_wrong_counts[key])
        print(f"Model_{key}: \t Correctly SO: {dict_correct_counts[key]} \t Wrong SO: {dict_wrong_counts[key]} \t TPR: {tpr:.4f}")
        if tpr <= 0.2:
            bad_ids.append(key)
    print(f"Bad ids: {bad_ids}")

# def compute_singling_out_acc_for_target_record(target_record, d, num_samples, meta_data, model_name, epsilon,
#                                                num_repetitions, output_path, sample_with_replacement=True,
#                                                start_random_state=42, max_workers=22):
#     result_list = get_singling_out_dataset(d, num_samples, meta_data, model_name, epsilon, num_repetitions,
#                                            output_path, sample_with_replacement, start_random_state, max_workers)
#     correct_singled_out = 0
#     counter = 0
#     for x, y in tqdm(result_list, desc="Performing singling out attack"):
#         if check_if_target_record_is_in_x(target_record, x):
#             correct_singled_out += 1
#         # if counter == 100:
#         #    break
#         # counter += 1
#     singling_out_acc = correct_singled_out / num_repetitions
#     print(f"Singling out accuracy: {singling_out_acc}")
#     return singling_out_acc


def sample_D_and_shadow_datasets(config):
    Path(config.output_path).mkdir(parents=True, exist_ok=True)
    D, D_sorted = get_D_and_sort_by_vuln(config.path_csv, config.size_d, randoms_state=42, target_path=config.output_path, convert_to_cat=True)
    #first_record = x.iloc[0]
    build_first_n_shadow_datasets(config.num_shadow_datasets, config.num_shadow_pairs, config.size_x, D_sorted,
                                  config.path_shadow_x, sample_with_replacement=config.sample_with_replacement,
                                  max_workers=config.max_workers)
    return D, D_sorted


def experiment_different_epsilons(config, epsilons):
    result_list = []
    out_put_path_parent = config.output_path
    for epsilon in epsilons:
        eps_str = str(epsilon).replace(".", "_")
        config.output_path = f"{out_put_path_parent}eps_new_{eps_str}_{config.synthetic_model}/"
        adjust_output_path_dependencies(config)
        Path(config.output_path).mkdir(parents=True, exist_ok=True)

        apply_m_to_multiple_shadow_datasets(config.num_shadow_datasets, config.num_shadow_pairs, config.n_synth,
                                            config.path_shadow_x, config.path_shadow_y, config.path_models, epsilon,
                                            config.synthetic_model, max_workers=config.max_workers,
                                            temp_path=config.temp_path,
                                            parallel=config.build_synthetic_models_in_parallel,
                                            n_synthetic_datasets=config.n_synthetic_datasets)

        attacks = ["black_box"]
        #attacks  = ["black_box"]
        first_run = True
        for attack in attacks:
            print(f"Performing {attack} attack")
            #path_attack_models = config.path_attack_models + "new_7/" + attack + "/"
            path_attack_models = config.path_attack_models + attack + "/"
            AttackModelContainer.build_and_train_attack_models(attack, config.num_shadow_datasets,
                                                               config.num_shadow_pairs, config.output_path,
                                                               config.path_shadow_x,
                                                               path_attack_models,
                                                               config.train_fraction,
                                                               max_workers=10,
                                                               #max_workers=25,
                                                               classifier_model=config.classifier_model)

            singling_out_acc, result_list, attack_model_container = compute_singling_out_acc(config.path_sampled_data, attack, config.synthetic_model,
                                                        epsilon, path_attack_models,
                                                        config.num_reps_for_so_acc, config.output_path, config.temp_path,
                                                        max_workers=config.max_workers, name="val", show_progress=True)
            analyse_result_so_list(result_list, attack_model_container.attack_models)
            with open(f"{config.output_path}Singling_out_accuracy.txt", 'a') as file:
                if first_run:
                    file.write("epsilon,attack_type,singling_out_acc\n")
                    first_run = False
                file.write(f"{epsilon},{attack},{singling_out_acc}\n")

            # singling_out_acc = compute_singling_out_acc_for_target_record(d_sorted_by_vuln,d, config.size_x,
            #                                                              meta_data, config.synthetic_model, epsilon,
            #                                            config.num_reps_for_so_acc,
            #                                            config.output_path,
            #                                            sample_with_replacement=config.sample_with_replacement,
            #                                            start_random_state=4000000000, max_workers=config.max_workers)
        #result_list.append({"epsilon": epsilon, "singling_out_acc": singling_out_acc})

    return
    # plot the results
    df = pd.DataFrame(result_list)
    df.to_csv(out_put_path_parent + "singling_out_acc_vs_epsilon.csv", index=False)
    plt.figure(figsize=(10, 6))
    plt.plot(df["epsilon"], df["singling_out_acc"], marker="x")
    plt.title("Singling-Out Accuracy vs. Epsilon")
    plt.xscale("log")
    plt.xlabel("Epsilon")
    plt.ylabel("Singling-Out Accuracy")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_put_path_parent + "singling_out_acc_vs_epsilon.png", dpi=400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform singling-out analysis")
    parser.add_argument(
        'config_path',
        #nargs='?',
        default="../config.yaml",
        type=str,
        help='The path to the config-file'
    )

    # Parse the arguments
    args = parser.parse_args()
    config = read_config(args.config_path)

    sample_data = True
    if sample_data:
        # The idea is to first sample the D and X which are used for building the shadow datasets if they are created
        # we set sample_data=False and apply m to the sampled datasets
        D, D_sorted = sample_D_and_shadow_datasets(config)
        sample_singling_out_datasets(D, config.size_x, config.num_reps_for_so_acc, target_path=config.output_path)
    else:
        experiment_different_epsilons(config, epsilons=[0.1, 1, 10, 100, 1000])
