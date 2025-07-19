from pathlib import Path

import arff
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tornado.options import parse_config_file

from src.utils.utils import read_config


def split_datasets_in_test_and_train(num_test, path_data, target_path, random_state=42, dataset="cardio"):
    if dataset == "cardio":
        path = f"{path_data}/cardio/cardio_cat.csv"
    elif dataset == "adult":
        path = f"{path_data}/adult/adult_cat.csv"
    elif dataset == "house":
        path = f"{path_data}/house/house.csv"

    df = pd.read_csv(path)
    if dataset == "cardio":
        df = df.rename(columns={'cardio': 'y'})
    if dataset == "adult":
        df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
        df = df.rename(columns={'income': 'y'})

        categorical_columns = df.select_dtypes(include=['object']).columns

        # Perform one-hot encoding on categorical columns
        encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' to avoid dummy variable trap
        encoded_categorical_data = encoder.fit_transform(df[categorical_columns])

        # Create a DataFrame with the encoded categorical data
        encoded_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_columns))

        # Concatenate the encoded columns with the original DataFrame (excluding the original categorical columns)
        df = pd.concat([df.drop(categorical_columns, axis=1), encoded_df], axis=1)


    # drop duplicates
    df = df.drop_duplicates()

    # split in train and test
    train_df, test_df = train_test_split(df, test_size=num_test, random_state=random_state)

    target_path_train = f"{target_path}{dataset}_train.csv"
    target_path_test = f"{target_path}{dataset}_test.csv"

    train_df.to_csv(target_path_train, index=False)
    test_df.to_csv(target_path_test, index=False)


def sample_dataset(path_data, num_samples=1000, random_state=42, dataset="cardio", target_path=None,
                   train_frac=0.8):

    df = pd.read_csv(f"{path_data}/{dataset}_train.csv")
    #print(len(df))
    df = df.sample(num_samples, random_state=random_state, replace=False)

    # reset index
    #df = df.reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=(1 - train_frac), random_state=random_state)

    if target_path:
        train_df.to_csv(target_path + f"{random_state}_train.csv", index=False)
        val_df.to_csv(target_path + f"{random_state}_val.csv", index=False)

    return train_df, val_df


def sample_multiple_datasets(path_data, num_datasets, min_samples, max_samples, target_path=None, seed=42):
    names_datasets = ["cardio", "adult", "house"]

    # set numpy seed
    np.random.seed(seed)

    for name in names_datasets:
        path = target_path + name + "/"
        Path(path).mkdir(parents=True, exist_ok=True)
        for i in range(num_datasets):
            # get random number of samples
            num_samples = np.random.randint(min_samples, max_samples)
            print(f"Sampling {num_samples} samples for dataset {i}")
            sample_dataset(path_data, num_samples, random_state=i, dataset=name, target_path=target_path + name + "/")

def read_sampled_dataset(path, id, split_in_y_and_x=True):
    df_train = pd.read_csv(path + f"{id}_train.csv")
    df_val = pd.read_csv(path + f"{id}_val.csv")
    #df_test = pd.read_csv(path + f"{id}_test.csv")

    if split_in_y_and_x:
        y_train = df_train[['y']]
        y_val = df_val[['y']]
        #y_test = df_test[['y']]

        x_train = df_train.drop(columns=['y'])
        x_val = df_val.drop(columns=['y'])
        #x_test = df_test.drop(columns=['y'])
        return x_train, y_train, x_val, y_val, #x_test, y_test
    else:
        return df_train, df_val#, df_test

def read_house_16_H_dataset(path, target_path=None):
    with open(path, 'r') as file:
        dataset = arff.load(file)

    # Convert to a pandas DataFrame
    df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])
    df = df.rename(columns={'price': 'y'})

    # save data
    if target_path:
        Path(target_path).mkdir(parents=True, exist_ok=True)
        df.to_csv(target_path + "house.csv", index=False)

    # rename price  column to y
    return df


if __name__ == "__main__":
    config_file = read_config()
    target_path = config_file.output_path + "/decision_tree_attack_data/"
    df = read_house_16_H_dataset(config_file.path_data + "house/house_16H.arff", target_path=config_file.path_data + "house/")
    print(df)
    split_datasets_in_test_and_train(1000, config_file.path_data, target_path, random_state=42, dataset="house")
    sample_multiple_datasets(config_file.path_decision_tree_attack_data, 100, 1000, 10000, target_path=target_path)