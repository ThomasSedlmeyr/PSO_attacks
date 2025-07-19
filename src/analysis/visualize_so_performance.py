import os


import pandas as pd
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.utils import read_config


# epsilon,attack_type,singling_out_acc
# 10,black_box,0.251
# 10,white_box,0.193


def get_data(model_name, folder_path, attack_type):
    folders = [str(f) for f in Path(folder_path).iterdir() if f.is_dir()]
    filtered = [f for f in folders if model_name in f and "_new_" in f and not "copy" in f]
    #filtered = [f for f in folders if model_name in f and not "_new_" n f and not "copy" in f and not "dpart" in fi]
    x, y = [], []
    for path in filtered:
        # check if path exists:
        if not Path(path + "/Singling_out_accuracy.txt").exists():
            print(f"Path {path} does not exist")
            continue
        df = pd.read_csv(path + "/Singling_out_accuracy.txt")
        df = df[df['attack_type'] == attack_type]
        x.append(df['epsilon'].to_numpy()[0])
        y.append(df['singling_out_acc'].to_numpy()[0])

    return x, y

def replace_words_in_heading(heading: str):
    # replace words in heading
    heading = heading.replace("dpart", "DPART")
    heading = heading.replace("syn_gen", "Synthetic Data Generation")
    heading = heading.replace("adult", "Adult Dataset")
    heading = heading.replace("cardio", "Cardio Dataset")

    return heading

def get_heading_and_file_name(folder_path):
    # get name of folder using pathlib
    path = Path(folder_path)
    folder_name = path.name
    splits = folder_name.split("_")
    if len(splits) == 3:
        result_heading = f'Singling Out Accuracy vs Epsilon for {splits[0]} |X| = {splits[1]} |D| = {splits[2]}'
        result_name = f"{splits[0]}_{splits[1]}_{splits[2]}_singling_out_accuracies.png"
    else:
        result_heading = "Singling Out Accuracy vs Epsilon"
        result_name = "singling_out_accuracies.png"
    return result_heading, result_name


def create_plot(folder_path, target_path):
    heading, result_name = get_heading_and_file_name(folder_path)
    heading = replace_words_in_heading(heading)
    # Assuming get_data returns two lists or arrays: epsilon and singling_out_acc
    epsilon1, singling_out_acc1 = get_data("dpart", folder_path, "black_box")
    epsilon2, singling_out_acc2 = get_data("syn_gen", folder_path, "black_box")
    #epsilon3, singling_out_acc3 = get_data("dpart", folder_path, "white_box")
    #epsilon4, singling_out_acc4 = get_data("syn_gen", folder_path, "white_box")

    # Create a DataFrame for plotting
    data = {
        'Epsilon': list(epsilon1) + list(epsilon2), #+ list(epsilon3) + list(epsilon4),
        'Singling Out Accuracy': list(singling_out_acc1) + list(singling_out_acc2), #+ list(singling_out_acc3) + list(singling_out_acc4),
        'Method': ['DPART'] * len(epsilon1) + 
                  ['Synthetic Data Generation'] * len(epsilon2) #+ 
                  #['dpart_white_box'] * len(epsilon3) + 
                  #['syn_gen_white_box'] * len(epsilon4)
    }
    df = pd.DataFrame(data)

    # Plotting using seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Epsilon', y='Singling Out Accuracy', hue='Method', marker='o')
    plt.xlabel('Epsilon')
    # logarithmic x-axis
    plt.xscale('log')
    # sometimes we have to invert the y-axis
    plt.gca().invert_yaxis()  # Invert the y-axis


    plt.ylabel('Singling Out Accuracy')
    plt.title(heading)
    plt.grid(True)
    plt.legend(title='Synthetic Data Generation Algorithm')
    target_file_path = f"{target_path}{result_name}"
    plt.tight_layout()
    plt.savefig(target_file_path, dpi=400)

if __name__ == "__main__":
    config = read_config()
    folder_path = config.output_path + "/adult_1000_5000/"
    target_path = config.output_path + "plots/"
    create_plot(folder_path, target_path)