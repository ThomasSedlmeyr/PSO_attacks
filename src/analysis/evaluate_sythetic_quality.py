import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency, entropy
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mutual_info_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from synthesis.evaluation.metrics import MarginalComparison, AssociationsComparison

from src.analysis.create_shadow_datasets import apply_m
from src.utils.utils import conv_to_cat, read_config

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def compare_datasets(original_df, synthetic_df):
    results = {}

    # Ensure the dataframes have the same columns
    assert original_df.columns.tolist() == synthetic_df.columns.tolist(), "DataFrames must have the same columns"

    for col in original_df.columns:
        orig_counts = original_df[col].value_counts(normalize=True)
        synth_counts = synthetic_df[col].value_counts(normalize=True)

        # Frequency Distribution Comparison using KL Divergence
        kl_divergence = entropy(orig_counts, synth_counts)
        results[f"KL Divergence for {col}"] = kl_divergence

        # Cramér’s V for association between categorical variables
        if len(original_df[col].unique()) > 1 and len(synthetic_df[col].unique()) > 1:
            contingency_table = pd.crosstab(original_df[col], synthetic_df[col])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            results[f"Cramér’s V for {col}"] = cramers_v

        # Mutual Information
        le = LabelEncoder()
        orig_encoded = le.fit_transform(original_df[col])
        synth_encoded = le.transform(synthetic_df[col])
        mi = mutual_info_score(orig_encoded, synth_encoded)
        results[f"Mutual Information for {col}"] = mi

    return results


def train_random_forest_adult(x_train, y_train, x_test, random_state=42):
    """
    Trains a Random Forest model on the given DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing features and target.
    target_column (str): The name of the target column.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Random state for reproducibility.

    Returns:
    clf (Pipeline): The trained model pipeline.
    train_score (float): Training set score.
    test_score (float): Test set score.
    """

    # concat x data in pandas
    X = pd.concat([x_train, x_test], axis=0)


    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()

    # Preprocessing pipelines for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create a pipeline that first preprocesses the data and then applies a classifier
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=random_state))
    ])

    # Fit the model
    clf.fit(x_train, y_train)
    predictions_train = clf.predict(x_train)
    predictions_test = clf.predict(x_test)

    return {"clf": clf, "pred_train": predictions_train, "pred_test": predictions_test}

def train_random_forest_cardio(x_train, y_train, x_test, random_state=42):
    clf = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=random_state)
    clf.fit(x_train, y_train)
    predictions_train = clf.predict(x_train)
    predictions_test = clf.predict(x_test)

    return {"clf": clf, "pred_train": predictions_train, "pred_test": predictions_test}

def evaluate_metrics(predictions_train, predictions_test, y_train, y_test):
    if len(np.unique(predictions_train)) == 2:
        auc_train = roc_auc_score(y_train, predictions_train)
    else:
        auc_train = 0
    if len(np.unique(predictions_test)) == 2:
        auc_test = roc_auc_score(y_test, predictions_test)
    else:
        auc_test = 0
    acc_train = accuracy_score(y_train, predictions_train)
    acc_test = accuracy_score(y_test, predictions_test)

    return {"train_auc": auc_train, "test_auc": auc_test, "train_acc": acc_train, "test_acc": acc_test}

def compute_metrics(x_original, y_original, x_synthetic, y_synthetic, categorical_data_only=False,
                    dataset_name="adult"):
    df_original = pd.concat([x_original, y_original], axis=1)
    df_synthetic = pd.concat([x_synthetic, y_synthetic], axis=1)
    metric_values = []
    marginal_comparison = MarginalComparison().fit(df_original, df_synthetic)
    score = marginal_comparison.score()
    metric_values.append(score)
    print(f"Jesen-Shannon distance: {score}")
    if not categorical_data_only:
        associations_comparison = AssociationsComparison().fit(df_original, df_synthetic)
        ass_score = associations_comparison.score()
        print(f"pairwise correlation distance: {ass_score}")
        metric_values.append(ass_score)


    x_train_original, x_test_original, y_train_original, y_test_original = (
        train_test_split(x_original, y_original, test_size=0.2, random_state=42))
    x_train_syn, x_test_syn, y_train_syn, y_test_syn = (
        train_test_split(x_synthetic, y_synthetic, test_size=0.2, random_state=42))

    train_function = None
    if dataset_name == "adult":
        train_function = train_random_forest_adult
    elif dataset_name == "cardio":
        train_function = train_random_forest_cardio
    else:
        raise ValueError("Dataset name unknown")

    original_result = train_function(x_train_original, y_train_original, x_test_original, random_state=42)
    metrics_original = evaluate_metrics(original_result["pred_train"], original_result["pred_test"], y_train_original, y_test_original)
    print(f"Original Train ACC: {metrics_original['train_acc']} AUC: {metrics_original['train_auc']}  ")
    print(f"Original Test ACC: {metrics_original['test_acc']} AUC: {metrics_original['test_auc']}  ")
    metric_values.append(metrics_original['test_auc'])

    result_syn = train_function(x_train_syn, y_train_syn, x_test_original, random_state=42)
    metrics_syn = evaluate_metrics(result_syn["pred_train"], result_syn["pred_test"], y_train_syn, y_test_original)
    print(f"Synthetic Train ACC: {metrics_syn['train_acc']} AUC: {metrics_syn['train_auc']}  ")
    print(f"Original Test ACC: {metrics_syn['test_acc']} AUC: {metrics_syn['test_auc']}  ")
    metric_values.append(metrics_syn['test_auc'])

    return metric_values

def get_adult_dataset(num_samples, random_state=42):
    path_csv = '/mnt/data/Dokumente/Studium/Master/Semester_4/Guided_Research/own_code/PSO_attacks/datasets/adult/adult_cat.csv'
    all_data = pd.read_csv(path_csv)
    all_data = all_data.head(num_samples)
    all_data['income'].replace({'<=50K': 0, '>50K': 1}, inplace=True)

    #if categorical_data_only:
    #    all_data = conv_to_cat(all_data)
    all_data = all_data.sample(n=num_samples, random_state=random_state, replace=False)
    return all_data

def separate_target(original, synthetic, target_name):
    # Separate features and target
    x_original = original.drop(columns=[target_name])
    y_original = original[target_name]
    x_synthetic = synthetic.drop(columns=[target_name])
    y_synthetic = synthetic[target_name]
    return x_original, y_original, x_synthetic, y_synthetic

def get_cardio_dataset(num_samples, random_state=42):
    path = "/mnt/data/Dokumente/Studium/Master/Semester_4/Guided_Research/own_code/PSO_attacks/datasets/cardio/cardio_cat.csv"
    df = pd.read_csv(path, delimiter=',')
    df = df.sample(n=num_samples, random_state=random_state, replace=False)
    return df

def analyse_different_different_synthetic_models(num_samples, epsilon=1.0, model_names=None, categorical_data_only=False, target_path="",
                                                 num_iterations=1, name="", dataset_name="adult"):
    target_name = "income" if dataset_name == "adult" else "cardio"
    if dataset_name == "adult":
        dataset_method = get_adult_dataset
    elif dataset_name == "cardio":
        dataset_method = get_cardio_dataset
    else:
        raise ValueError("Dataset name unknown")


    #models = ["privbayes_data_syn", "privbayes_syn_gen", "marginal", "privbayes_dpart"]
    #models = ["synthpop"]
    results = {model: [] for model in model_names}

    for i in range(num_iterations):
        seed = i * 10 + 42  # Different seeds for each run
        original_data = dataset_method(num_samples, random_state=seed)
        if categorical_data_only:
            original_data = conv_to_cat(original_data)
        for model in model_names:
            print("Model: ", model, "Iteration: ", i)
            synthetic_df, fitted_model = apply_m(original_data, meta_data=None, output_path=None,
                                                 num_syn_samples=num_samples,
                                                 epsilon=epsilon, model_name=model, seed=seed,
                                                 categorical_data_only=categorical_data_only)
            x_original, y_original, x_synthetic, y_synthetic = separate_target(original_data, synthetic_df[0], target_name)
            metrics = compute_metrics(x_original, y_original, x_synthetic, y_synthetic,
                                      categorical_data_only=categorical_data_only, dataset_name=dataset_name)
            results[model].append(metrics)

    # Average the results
    # and round to 4 decimal places

    avg_results = {model: np.mean(results[model], axis=0).round(4) for model in model_names}

    if categorical_data_only:
        metric_names = ["Jensen-Shannon distance", "Original Train score", "Synthetic Test score"]
    else:
        metric_names = ["Jensen-Shannon distance", "pairwise correlation distance", "Original Train score",
                        "Synthetic Test score"]

    # Convert results to a DataFrame
    avg_results_df = pd.DataFrame(avg_results, index=metric_names).T

    # Save the results as an image
    plt.figure(figsize=(10, 2))
    # set the title of the table
    plt.title(name)
    plt.table(cellText=avg_results_df.values, colLabels=avg_results_df.columns, rowLabels=avg_results_df.index,
              loc='center')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(target_path + name + ".png", dpi=300)

    return avg_results_df

def perform_multiple_experiments(model_names, dataset_names, epsilons, num_samples, target_path,
                                 categorical_data_only=True, num_iterations=1):

    for dataset_name in dataset_names:
        result_list = []
        for epsilon in epsilons:
            print(f"Dataset: {dataset_name} Epsilon: {epsilon} Num_samples: {num_samples} Categorical: {categorical_data_only}")
            name = f"new_{num_iterations}_iterations_{dataset_name}_eps_{epsilon}_num_{num_samples}_categorical_{categorical_data_only}"
            result = analyse_different_different_synthetic_models(num_samples=num_samples, epsilon=epsilon,
                                                             model_names=model_names,
                                                             categorical_data_only=categorical_data_only,
                                                             target_path=target_path, num_iterations=num_iterations,
                                                             name=name, dataset_name=dataset_name)
            result_list.append((epsilon, result))
        generate_line_plot(result_list, dataset_name, target_path, model_names)
def generate_line_plot(result_list, data_set_name, target_path, model_names):
    plt.figure(figsize=(10, 10))
    first_data_frame = result_list[0][1]

    # Create an empty DataFrame to store the transformed data
    combined_df = pd.DataFrame()

    for epsilon, df in result_list:
        # Flatten the DataFrame
        flat_df = df.unstack().reset_index()
        flat_df.columns = ['Column', 'Row', 'Value']
        flat_df['Epsilon'] = epsilon  # This column will help in reshaping

        # Set the index to Row and Column for joining
        flat_df.set_index(['Row', 'Column'], inplace=True)
        flat_df.drop('Epsilon', axis=1, inplace=True)  # Drop Epsilon column, not needed now
        flat_df.rename(columns={'Value': epsilon}, inplace=True)  # Rename 'Value' to epsilon value

        # Combine with the main DataFrame
        if combined_df.empty:
            combined_df = flat_df
        else:
            # Use join to add new epsilon column to combined_df
            combined_df = combined_df.join(flat_df, how='outer')



    index = pd.MultiIndex.from_product([model_names,
                                        ['Jensen-Shannon distance', 'Original Train score', 'Synthetic Test score']],
                                       names=['Row', 'Column'])
    df = pd.DataFrame(combined_df, index=index)

    distances_df = df.xs('Jensen-Shannon distance', level='Column')
    distances_df.rename(index={'privbayes_dpart': 'DPART'}, level='Row', inplace=True)
    distances_df.rename(index={'privbayes_syn_gen': 'Synthetic Data Generation'}, level='Row', inplace=True)
    scores_df = df[df.index.get_level_values('Column').str.contains('score')]
    scores_df.rename(index={'privbayes_dpart': 'DPART'}, level='Row', inplace=True)
    scores_df.rename(index={'privbayes_syn_gen': 'Synthetic Data Generation'}, level='Row', inplace=True)

    print(scores_df)

    # Plotting distances
    plt.figure(figsize=(10, 6))
    for row in ['DPART', 'Synthetic Data Generation']:
        plt.plot(distances_df.columns, distances_df.loc[row], marker='o', label=row)
    name = data_set_name
    name = name.replace("adult", "Adult")
    name = name.replace("cardio", "Cardio")
    plt.title(f'Jensen-Shannon Distances for {name} Dataset')
    plt.xlabel('Epsilon')
    plt.xscale('log')
    plt.ylabel('Distance')
    plt.legend(title='Synthetic Data Generation Algorithm')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(target_path + f"distances_{data_set_name}.png", dpi=400)
    #plt.show()

    # Plotting scores
    plt.figure(figsize=(10, 6))
    was_original = False
    names = [('DPART', 'Synthetic Test score'),
            ('Synthetic Data Generation', 'Synthetic Test score'),
            ('Synthetic Data Generation', 'Original Train score')]
    for row in names:
        # The original test score is for each model the same, so we need to plot it only once
        if 'Original' in row[1] :
            if not was_original:
                plt.plot(scores_df.columns, scores_df.loc[row], marker='o', label="Original Data")
                was_original = True
        else:
            name = f"{row[1]}"
            name = name.replace("Synthetic Test score", "")
            plt.plot(scores_df.columns, scores_df.loc[row], marker='o', label=f"{row[0]} {name}")

    name = data_set_name
    name = name.replace("adult", "Adult")
    name = name.replace("cardio", "Cardio")
    plt.title(f'Classifier AUC for {name} Dataset')
    plt.xlabel('Epsilon')
    # log scale for better visualization
    plt.xscale('log')
    plt.ylabel('AUC')
    plt.legend(title='Synthetic Data Generation Algorithm', loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(target_path + f"Classifier_AUCs_{data_set_name}.png", dpi=400)

if __name__ == "__main__":
    num_samples = 1000
    categorical_data_only = True
    dataset_name = "adult"
    model_names = ["privbayes_syn_gen", "privbayes_dpart"]
    config = read_config()
    dataset_names = ["cardio", "adult"]
    epsilons = [0.1, 1, 10, 100, 1000]
    perform_multiple_experiments(model_names, dataset_names, epsilons, num_samples, config.output_path,
                                 categorical_data_only=True, num_iterations=1)