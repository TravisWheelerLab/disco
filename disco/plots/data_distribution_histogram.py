import os
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics

from disco.inference_utils import load_pickle

"""
This script is intended to reproduce the figure in the paper that describes the train/test/validation class
distributions.
"""

class_code_to_name = {0: "A", 1: "B", 2: "BACKGROUND"}


def count_event_and_pointwise_data():
    data_directories = ["train", "test", "validation"]
    datasets_dict = {}
    root = "/xdisk/twheeler/colligan/disco/extracted_data_1150_nffts"

    eventwise = {
        "test": defaultdict(int),
        "train": defaultdict(int),
        "validation": defaultdict(int),
    }
    pointwise = {
        "test": defaultdict(int),
        "train": defaultdict(int),
        "validation": defaultdict(int),
    }

    for data_dir in data_directories:
        files = glob(os.path.join(root, data_dir, "*.pkl"))
        labels = [load_pickle(pkl)[1] for pkl in files]

        for label_array in labels:
            unique_labels, counts = np.unique(label_array, return_counts=True)
            for label, count in zip(unique_labels, counts):
                pointwise[data_dir][label] += count

            if len(unique_labels) == 1:
                eventwise[data_dir][unique_labels[0]] += 1
            else:
                # where there's a transition in labels, the difference is _not_ equal to 0.
                # by indexing into the label array where the class label transitions from one to another,
                # we can get the identity of each class.
                classes = label_array[1:][np.diff(label_array) != 0]
                # we also have to add the first class in, since we're dealing with differences, and the first
                # transition (from "NULL" to the first class present in the label array) won't be accounted for.
                eventwise[data_dir][label_array[0]] += 1
                for c in classes:
                    eventwise[data_dir][c] += 1

    pointwise_df = pd.DataFrame.from_dict(pointwise)
    eventwise_df = pd.DataFrame.from_dict(eventwise)

    pointwise_df.to_csv("pointwise_counts.csv")
    eventwise_df.to_csv("eventwise_counts.csv")


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    pointwise = pd.read_csv(os.path.join(root, "pointwise_counts.csv")).rename(
        columns={"Unnamed: 0": "class code"}
    )
    eventwise = pd.read_csv(os.path.join(root, "eventwise_counts.csv")).rename(
        columns={"Unnamed: 0": "class code"}
    )
    fig, ax = plt.subplots(ncols=2)

    sns.barplot(pointwise, x="class code", y="test", ax=ax[0])
    sns.barplot(pointwise, x="class code", y="train", ax=ax[0])
    sns.barplot(pointwise, x="class code", y="validation", ax=ax[0])

    sns.barplot(evetnwise, x="class code", y="test", ax=ax[0])
    sns.barplot(evetnwise, x="class code", y="train", ax=ax[0])
    sns.barplot(evetnwise, x="class code", y="validation", ax=ax[0])
