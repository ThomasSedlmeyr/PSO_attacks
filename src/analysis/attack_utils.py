from pathlib import Path
import dill
import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
from xgboost import XGBClassifier

from src.analysis.querybased import get_queries, extract_querybased
from src.synthetic_models import model_container
from src.utils.utils import conv_to_cat


def perform_hyper_parameter_optimization_xgb(x, y):
    param_dist = {
        'max_depth': stats.randint(3, 8),
        'learning_rate': stats.uniform(0.01, 0.1),
        'subsample': stats.uniform(0.5, 0.5),
        'n_estimators':stats.randint(50, 300),
        'min_child_weight': stats.randint(1, 10),

    }

    # Create the XGBoost model object
    xgb_model = XGBClassifier()

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=40, cv=5, scoring='accuracy', n_jobs=40, random_state=42,)

    # Fit the RandomizedSearchCV object to the training data
    random_search.fit(x, y)

    # Print the best set of hyperparameters and the corresponding score
    print("Best set of hyperparameters: ", random_search.best_params_)
    print("Best score: ", random_search.best_score_)

    best_clf = random_search.best_estimator_
    return best_clf


def perform_hyper_parameter_optimization(x, y):
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=1)

    # Define the parameter grid
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'bootstrap': [True, False]
    }

    # Set up the randomized search with cross-validation
    random_search = RandomizedSearchCV(
        clf, param_distributions=param_dist, n_iter=50, cv=5, verbose=1, random_state=42, n_jobs=1, scoring='accuracy'
    )

    print("Performing Randomized Search...")
    # Fit the randomized search model
    random_search.fit(x, y)

    # Get the best parameters and estimator
    best_params = random_search.best_params_
    best_clf = random_search.best_estimator_

    print("Best Parameters:", best_params)
    print("Best Estimator:", best_clf)

    return best_clf

def extract_features_bb(synth_df, target_record, full_uniq_vals=None, queries=None, attack_type='querybased'):
    # extract black box features from synthetic dataset
    if attack_type == 'querybased':
        feat = extract_querybased(target_record, synth_df, queries)
    else:
        raise ValueError("Invalid attack type")
    return feat


def train_clf_and_attack(feats_in, feats_out, num_train, target_record, target_path=None, model="gradient_boosting",
                         path_roc_curve=None, additional_test=False):
    # build classifier using first `n_shadow` feats and run inference on remaining feats
    if additional_test:
        num_val = int((len(feats_in) - num_train) // 2)
        num_test = int(len(feats_in) - num_train - num_val)
    else:
        num_val = len(feats_in) - num_train
        num_test = 0

    # trim feats to `n_shadow` each for `in` and `out` worlds
    train_feats = np.concatenate([feats_in[:num_train], feats_out[:num_train]])
    train_labels = np.array([1] * len(feats_in[:num_train]) + [0] * len(feats_out[:num_train]))


    # train classifier on `n_shadow` feats each for `in` and `out` worlds
    if model == "gradient_boosting":
        clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        #clf = perform_hyper_parameter_optimization_xgb(train_feats, train_labels)
    elif model == "random_forest":
        clf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1)
    elif model == "extra_trees":
        # clf = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=1)
        clf = perform_hyper_parameter_optimization(train_feats, train_labels)
    else:
        raise ValueError("Invalid model type")

    #print("Fitting Classifier...")
    #clf.fit(train_feats, train_labels)
    #print("Classifier trained")
    mia_scores_train = clf.predict_proba(train_feats)[:, 1]
    train_auc = roc_auc_score(train_labels, mia_scores_train)

    # run inference on val
    split_index = num_train + num_val
    features_val = np.concatenate([feats_in[num_train:split_index], feats_out[num_train:split_index]])
    val_labels = np.array([1] * num_val + [0] * num_val)
    mia_scores_val = clf.predict_proba(features_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(val_labels, mia_scores_val)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[f1_scores.argmax()]

    binary_prediction_val = (mia_scores_val >= best_threshold).astype(int)
    acc_val = accuracy_score(val_labels, binary_prediction_val)
    auc_val = roc_auc_score(val_labels, mia_scores_val)
    partial_auc_val = roc_auc_score(val_labels, mia_scores_val, max_fpr=0.05)
    plot_and_save_roc_curve(val_labels, mia_scores_val, target_path=path_roc_curve)
    print(f"Train AUC: {train_auc} Val ACC: {acc_val}, Val AUC: {auc_val}, Val Partial AUC: {partial_auc_val}")

    if additional_test:
        # run inference on test
        features_test = np.concatenate([feats_in[split_index:], feats_out[split_index:]])
        test_labels = np.array([1] * num_test + [0] * num_test)
        mia_scores_test = clf.predict_proba(features_test)[:, 1]

        binary_prediction_test = (mia_scores_test >= best_threshold).astype(int)
        acc_test = accuracy_score(test_labels, binary_prediction_test)
        auc_test = roc_auc_score(test_labels, mia_scores_test)
        partial_auc_test = roc_auc_score(test_labels, mia_scores_test, max_fpr=0.05)
        print(f"Train AUC: {train_auc} Test ACC: {acc_test}, Test AUC: {auc_test}, Test Partial AUC: {partial_auc_test}")
        plot_and_save_roc_curve(test_labels, mia_scores_test, target_path=path_roc_curve)

    return clf, best_threshold, acc_val, auc_val, partial_auc_val, target_record, len(feats_in[0])

def compute_wb_features(models_in, models_out):
    features_in = [model.get_wb_features() for model in models_in]
    features_out = [model.get_wb_features() for model in models_out]
    return features_in, features_out


def compute_bb_features(synth_dfs_in, synth_dfs_out, metadata, target_record):
    df_cols = list(synth_dfs_in[0].columns)
    full_uniq_vals = {cdict['name']: cdict['i2s'] for cdict in metadata['columns']}

    # extract features
    queries = get_queries(df_cols)
    feats_in = [extract_features_bb(synth_df, target_record, full_uniq_vals, queries=queries)
                for synth_df in synth_dfs_in]
    feats_out = [extract_features_bb(synth_df, target_record, full_uniq_vals, queries=queries)
                 for synth_df in synth_dfs_out]
    return feats_in, feats_out


def load_shadow_dataset(shadow_size, path_shadow_ds):
    with open(path_shadow_ds, 'rb') as file:
        pair_list = joblib.load(file)

    data_in, data_out = [], []
    for pair in pair_list:
        data_in.append(pair[0])
        data_out.append(pair[1])

    return data_in[:shadow_size], data_out[:shadow_size]

def load_shadow_models(shadow_size, path_shadow_models):
    result_dict = model_container.load(path_shadow_models)
    #result_dict = dill.load(open(path_shadow_models, 'rb'))
    return result_dict["models_in"][:shadow_size], result_dict["models_out"][:shadow_size]

def load_target_record_and_meta_data(path_sampled_shadow_data_set_pair):
    """
    Loads the target record from a data pair. As the target record is always the last record in the df_in dataset. 
    """
    with open(path_sampled_shadow_data_set_pair, 'rb') as file:
        data = dill.load(file)
        df_in = data["df_in"]
        meta_data = data["metadata"]
        target_record = conv_to_cat(df_in).iloc[[-1]]
    return target_record, meta_data


def load_target_record_and_meta_data_2(path_x_dataset):
    """
    Loads the target record from a data pair. As the target record is always the last record in the df_in dataset.
    """
    with open(path_x_dataset, 'rb') as file:
        data = joblib.load(file)
        df_in = data[0]["df_in"]
        meta_data = data[0]["metadata"]
        target_record = df_in.iloc[[-1]]
    return target_record, meta_data

def train_attack_model(attack_type, num_shadow_pairs, path_saved_data, path_shadow_x, target_path,
                       num_train, index=0, path_roc_curve=None, classifier_type="gradient_boosting"):
    Path(target_path).mkdir(parents=True, exist_ok=True)
    print("Loading X....")
    path_x_concrete_shadow_ds = f"{path_shadow_x}/{index}_x_in_out_meta_dataset.joblib"
    target_record, meta_data = load_target_record_and_meta_data_2(path_x_concrete_shadow_ds)

    if attack_type == 'black_box':
        print("Loading Y...")
        path_y_concrete_shadow_ds = f"{path_saved_data}shadow_y/{index}_y_in_out_dataset.joblib"
        print("Loading Shadow Dataset...")
        synth_dfs_in, synth_dfs_out = load_shadow_dataset(num_shadow_pairs, path_y_concrete_shadow_ds)
        print("Computing black_box features...")
        feats_in, feats_out = compute_bb_features(synth_dfs_in, synth_dfs_out, meta_data, target_record)
    elif attack_type == 'white_box':
        path_shadow_models = f"{path_saved_data}models/{index}_models_in_and_out.dill"
        print("Loading shadow models...")
        models_in, models_out = load_shadow_models(num_shadow_pairs, path_shadow_models)
        print("Computing white box features...")
        feats_in, feats_out = compute_wb_features(models_in, models_out)
        # normalize all features to equal length
        max_feat_len = max({len(feat) for feat in feats_in + feats_out})
        feats_in = [feat + [0] * (max_feat_len - len(feat)) for feat in feats_in]
        feats_out = [feat + [0] * (max_feat_len - len(feat)) for feat in feats_out]
    else:
        raise ValueError("Invalid attack type")

    print("Building Classifier...")
    result_tuple = train_clf_and_attack(feats_in, feats_out, num_train, target_record, f"{target_path}/{index}",
                                        classifier_type, path_roc_curve=path_roc_curve)
    return result_tuple

def train_attack_models(num_shadow_datasets, num_shadow_pairs, path_saved_data,
                        target_path, num_train, path_roc_curve=None):
    for i in tqdm(range(num_shadow_datasets), "Training and Tuning Attack Models"):
        train_attack_model(num_shadow_pairs, path_saved_data, target_path,
                           num_train, index=i, path_roc_curve=path_roc_curve)


def plot_and_save_roc_curve(y_true, y_pred_prob, target_path=None):
    """
    Plots the ROC curve and saves the image to the specified target path.

    Parameters:
    - y_true: Ground truth (correct) target values.
    - y_pred_prob: Estimated probabilities or decision function.
    - target_path: The path where the ROC curve image will be saved.
    """

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    sns.set(style='whitegrid')  # Set the Seaborn style
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = 'Receiver Operating Characteristic (ROC) Curve'
    if target_path is not None:
        # split from last slah to the end from target_path
        title = target_path.split("/")[-1]
    plt.title(title)
    plt.legend(loc='lower right')

    # Save the plot to the specified file
    if target_path is not None:
        plt.savefig(target_path)
    #plt.show()
    plt.close()

if __name__ == '__main__':
    shadow_size = 1000
    num_syn_samples = 1000
    num_shadow_pairs = 100
    num_train = 80
    num_shadow_datasets = 5

    path_synthetic_shadow_ds = "/mnt/data/Dokumente/Studium/Master/Semester_4/Guided_Research/own_code/PSO_attacks/output/first_10_shadow_datasets/m/"
    path_sampled_shadow_data_sets = "/mnt/data/Dokumente/Studium/Master/Semester_4/Guided_Research/own_code/PSO_attacks/output/first_10_shadow_datasets/samples/"
    target_path = "/mnt/data/Dokumente/Studium/Master/Semester_4/Guided_Research/own_code/PSO_attacks/output/first_10_shadow_datasets/attacks/"
    train_attack_models(num_shadow_datasets, num_shadow_pairs, path_synthetic_shadow_ds, path_sampled_shadow_data_sets, target_path, num_train)