import seaborn as sns

sns.set_palette(sns.color_palette("crest"))

import matplotlib.pyplot as plt
import pandas as pd

"""
This script is intended to reproduce the point-wise accuracy figure in the manuscript.
It shows the recall/precision of the 10-member ensemble as a function of iqr threshold.
"""


# this might be a little annoying.
# it's probably going to be easier to just not use the plotter object.
results_file = f"disco/resources/disco_accuracy_csvs/snr_0_ensemble_10_random_init/point_wise_10_random_init.csv"
df = pd.read_csv(results_file, index_col=0)
df.loc["iqr threshold"] = [float(colname.split("/")[0]) for colname in df.columns]
iqr_thresholds = pd.to_numeric(df.loc["iqr threshold"])

a_recall = pd.to_numeric(df.loc["recall, A chirp"])
b_recall = pd.to_numeric(df.loc["recall, B chirp"])
a_precision = pd.to_numeric(df.loc["precision, A chirp"])
b_precision = pd.to_numeric(df.loc["precision, B chirp"])

dataframe = pd.DataFrame.from_dict(
    {
        "iqr": iqr_thresholds,
        "A precision": a_precision,
        "A recall": a_recall,
        "B precision": b_precision,
        "B recall": b_recall,
    }
)

# remove rows where the results were thresholded by votes
dataframe = dataframe.drop(labels=["1/6", "1/7", "1/8", "1/9"], axis=0)

fig, ax = plt.subplots()
sns.lineplot(data=dataframe, x="iqr", y="A precision", label="A precision", ax=ax)
sns.lineplot(data=dataframe, x="iqr", y="A recall", label="A recall", ax=ax)
sns.lineplot(data=dataframe, x="iqr", y="B precision", label="B precision", ax=ax)
sns.lineplot(data=dataframe, x="iqr", y="B recall", label="B recall", ax=ax)

sns.scatterplot(data=dataframe, x="iqr", y="A precision", ax=ax)
sns.scatterplot(data=dataframe, x="iqr", y="A recall", ax=ax)
sns.scatterplot(data=dataframe, x="iqr", y="B precision", ax=ax)
sns.scatterplot(data=dataframe, x="iqr", y="B recall", ax=ax)

ax.invert_xaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color("#808080")
ax.spines["left"].set_color("#808080")
alpha = 0.3

ax.grid(alpha=alpha)
ax.set_xlabel("iqr threshold")
ax.set_ylabel("recall/precision")
ax.set_title("pointwise accuracy and iqr threshold")

plt.show()

# iqr threshold, votes, and percent coverage event wise
