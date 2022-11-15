import seaborn as sns

sns.set_palette(sns.color_palette("crest"))

import matplotlib.pyplot as plt
import pandas as pd

"""
This script is intended to compare the randomly initialized and bootstrapped ensembles.
"""

iqrs = [1, 0.4, 0.2, 0.1, 0.05]


def plot_b_f1(ax):
    results = []
    for method in ["bootstrap", "random_init"]:
        for ensemble_size in [2, 10, 30]:
            results_file = f"disco/resources/disco_accuracy_csvs/snr_0/point_wise_{ensemble_size}_{method}.csv"
            results_df = pd.read_csv(results_file, index_col=0)
            for iqr in iqrs:
                recall = float(results_df[f"{iqr}/0"]["recall, B chirp"])
                precision = float(results_df[f"{iqr}/0"]["precision, B chirp"])
                f1 = (2 * recall * precision) / (recall + precision)
                results_dict = {
                    "members": ensemble_size,
                    "method": method,
                    "iqr": iqr,
                    "f1": f1,
                }
                results.append(results_dict)

    results = pd.DataFrame.from_dict(results)

    sns.lineplot(
        data=results,
        x="iqr",
        y="f1",
        style="method",
        hue="members",
        palette="crest",
        ax=ax,
        legend=None,
    )
    sns.scatterplot(
        data=results,
        x="iqr",
        y="f1",
        style="method",
        hue="members",
        palette="crest",
        ax=ax,
        legend=None,
    )


def plot_a_f1(ax):
    results = []
    for method in ["bootstrap", "random_init"]:
        for ensemble_size in [2, 10, 30]:
            results_file = f"disco/resources/disco_accuracy_csvs/snr_0/point_wise_{ensemble_size}_{method}.csv"
            results_df = pd.read_csv(results_file, index_col=0)
            for iqr in iqrs:
                recall = float(results_df[f"{iqr}/0"]["recall, A chirp"])
                precision = float(results_df[f"{iqr}/0"]["precision, A chirp"])
                f1 = (2 * recall * precision) / (recall + precision)
                results_dict = {
                    "members": ensemble_size,
                    "method": method,
                    "iqr": iqr,
                    "f1": f1,
                }
                results.append(results_dict)

    results = pd.DataFrame.from_dict(results)

    sns.lineplot(
        data=results,
        x="iqr",
        y="f1",
        style="method",
        hue="members",
        palette="crest",
        ax=ax,
        legend=None,
    )
    sns.scatterplot(
        data=results,
        x="iqr",
        y="f1",
        style="method",
        hue="members",
        palette="crest",
        ax=ax,
        legend=None,
    )


def plot_accuracy(ax):
    results = []
    for method in ["bootstrap", "random_init"]:
        for ensemble_size in [2, 10, 30]:
            results_file = f"disco/resources/disco_accuracy_csvs/snr_0/point_wise_{ensemble_size}_{method}.csv"
            results_df = pd.read_csv(results_file, index_col=0)
            for iqr in iqrs:
                results_dict = {
                    "members": ensemble_size,
                    "method": method,
                    "iqr": iqr,
                    "accuracy": float(results_df[f"{iqr}/0"]["accuracy"]),
                }
                results.append(results_dict)

    results = pd.DataFrame.from_dict(results)

    sns.lineplot(
        data=results,
        x="iqr",
        y="accuracy",
        style="method",
        hue="members",
        palette="crest",
        ax=ax,
        legend=None,
    )
    sns.scatterplot(
        data=results,
        x="iqr",
        y="accuracy",
        style="method",
        hue="members",
        palette="crest",
        ax=ax,
    )


fig, ax = plt.subplots(ncols=3)
plot_accuracy(ax[0])
plot_a_f1(ax[1])
plot_b_f1(ax[2])

alpha = 0.3

for a in ax:
    a.invert_xaxis()
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.spines["bottom"].set_color("#808080")
    a.spines["left"].set_color("#808080")
    a.grid(alpha=alpha)

ax[1].legend(loc="lower left")
ax[1].set_ylabel("f1-score")
ax[2].set_ylabel("f1-score")
ax[0].set_ylim(0.5, 0.9)
ax[1].set_ylim(0.5, 0.9)
ax[2].set_ylim(0.5, 0.9)

ax[1].yaxis.set_ticklabels([])
ax[2].yaxis.set_ticklabels([])

plt.suptitle("bootstrapping vs random init performance, pointwise accuracy")


plt.show()
