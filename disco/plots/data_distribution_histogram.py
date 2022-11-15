import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

sns.set_palette(sns.color_palette("crest"))
import os
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from disco.inference_utils import load_pickle

"""
This script is intended to reproduce the figure in the paper that describes the train/test/validation class
distributions.
"""

class_code_to_name = {0: "A", 1: "B", 2: "bg"}


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.2f"  # Give format here


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
    # move class codes to names
    for class_code, name in class_code_to_name.items():
        eventwise["class code"].loc[eventwise["class code"] == class_code] = name
        pointwise["class code"].loc[eventwise["class code"] == class_code] = name

    fig, ax = plt.subplots(ncols=3, nrows=2, sharex=True)

    sns.barplot(pointwise, x="class code", y="train", ax=ax[0, 0])
    sns.barplot(pointwise, x="class code", y="test", ax=ax[0, 1])
    sns.barplot(pointwise, x="class code", y="validation", ax=ax[0, 2])

    sns.barplot(eventwise, x="class code", y="train", ax=ax[1, 0])
    sns.barplot(eventwise, x="class code", y="test", ax=ax[1, 1])
    sns.barplot(eventwise, x="class code", y="validation", ax=ax[1, 2])

    # ax[0].spines["top"].set_visible(False)
    # ax[0].spines["right"].set_visible(False)
    # ax[0].spines["bottom"].set_color("#808080")
    # ax[0].spines["left"].set_color("#808080")

    # ax[1].spines["top"].set_visible(False)
    # ax[1].spines["right"].set_visible(False)
    # ax[1].spines["bottom"].set_color("#808080")
    # ax[1].spines["left"].set_color("#808080")

    # ax[0].set_xlabel("")
    # ax[1].set_xlabel("")

    # ax[0].set_ylabel("count")
    # ax[1].set_ylabel("")

    # ax[0].set_title("pointwise labels")
    # ax[1].set_title("eventwise labels")

    for row_idx in range(2):
        for col_idx in range(3):
            # make the formatting the same, found on stackoverflow:
            # https://stackoverflow.com/questions/49351275/matplotlib-use-fixed-number-of-decimals-with-scientific-notation-in-tick-labels
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0, 0))
            ax[row_idx, col_idx].yaxis.set_major_formatter(yfmt)

            ax[row_idx, col_idx].ticklabel_format(
                style="sci", axis="y", scilimits=(0, 0)
            )
            ax[row_idx, col_idx].spines["top"].set_visible(False)
            ax[row_idx, col_idx].spines["right"].set_visible(False)
            ax[row_idx, col_idx].spines["bottom"].set_color("#808080")
            ax[row_idx, col_idx].spines["left"].set_color("#808080")
            ax[row_idx, col_idx].set_xlabel("")
            ax[row_idx, col_idx].set_xlabel("")
            ax[row_idx, col_idx].set_ylabel("count")
            ax[row_idx, col_idx].set_ylabel("")
            # ax[row_idx, col_idx].yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

    # make the y-axes the same for the different types: eventwise and pointwise
    ax[0, 0].sharey(ax[0, 1])
    ax[0, 1].sharey(ax[0, 2])
    ax[0, 1].autoscale()

    ax[1, 0].sharey(ax[1, 1])
    ax[1, 1].sharey(ax[1, 2])
    ax[1, 1].autoscale()

    # remove redundant ticks
    ax[0, 1].set_yticks([])
    ax[0, 2].set_yticks([])
    ax[1, 1].set_yticks([])
    ax[1, 2].set_yticks([])

    ax[0, 0].set_title("train", fontsize=10)
    ax[0, 1].set_title("test", fontsize=10)
    ax[0, 2].set_title("validation", fontsize=10)

    ax[0, 0].set_ylabel("pointwise")
    ax[1, 0].set_ylabel("eventwise")

    fig.text(
        y=0.5,
        x=0.02,
        fontsize=15,
        s="count",
        rotation="vertical",
        horizontalalignment="center",
        verticalalignment="center",
    )
    fig.text(
        y=0.02,
        x=0.5,
        fontsize=15,
        s="class",
        rotation="horizontal",
        horizontalalignment="center",
        verticalalignment="center",
    )

    plt.suptitle("point and event-wise distributions of classes", fontsize=15)

    plt.show()
