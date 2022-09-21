import sklearn.metrics
import disco.inference_utils as infer
import pandas as pd
from disco.extract_data import convert_time_to_index, w2s_idx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
from glob import glob
from scipy import stats
import disco.inference_utils as infer
from disco.dataset import SpectrogramDatasetMultiLabel
class_code_to_name = {0: "A", 1: "B", 2: "BACKGROUND"}


data_directories = ["train", "test", "validation"]
datasets_dict = {}

for data_dir in data_directories:
    data_path = os.path.join("..", "data", "final_data_extract", data_dir)
    files = glob(os.path.join(data_path, "*.pkl"))

    labels = [infer.load_pickle(pkl)[1] for pkl in files]

    all_labels = []
    for label_array in labels:
        unique_labels = np.unique(label_array)
        all_labels.extend(unique_labels)
    unique, counts = np.unique(all_labels, return_counts=True)

    datasets_dict[data_dir] = {}

    for i in range(len(unique)):
        datasets_dict[data_dir][class_code_to_name[unique[i]]] = counts[i]

for data_dir in data_directories:
    plt.bar(datasets_dict[data_dir].keys(), datasets_dict[data_dir].values(), width=0.5, color='cadetblue')
    plt.title("Sound event occurrences for " + data_dir + " set")
    plt.show()
    plt.close()

