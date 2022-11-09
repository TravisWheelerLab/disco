import seaborn as sns

sns.set_palette(sns.color_palette("crest"))

import matplotlib.pyplot as plt
import pandas as pd

"""
This script reproduces the figure showing event-wise accuracy as a function of IQR.
"""
proportion_correct = 90

results_file = f"disco/resources/disco_accuracy_csvs/snr_80_ensemble_10_random_init/event_wise_10_random_init.csv"
df = pd.read_csv(results_file, index_col=0)
df.loc["iqr threshold"] = [float(colname.split("/")[0]) for colname in df.columns]
df.loc["votes"] = [float(colname.split("/")[1]) for colname in df.columns]
df.loc["proportion correct"] = [float(colname.split("/")[2]) for colname in df.columns]

# remove the columns that were used to ablate over number of votes
non_voting_cols = df.loc["votes"] == 0
df = df.loc[:, non_voting_cols]
# remove the columns in proportion correct
proportion_correct_cols = df.loc["proportion correct"] == proportion_correct
df = df.loc[:, proportion_correct_cols]

iqr_thresholds = pd.to_numeric(df.loc["iqr threshold"])

a_recall = pd.to_numeric(df.loc["recall, A chirp"])
b_recall = pd.to_numeric(df.loc["recall, B chirp"])
overall = pd.to_numeric(df.loc["accuracy"])


dataframe = pd.DataFrame.from_dict(
    {
        "iqr": iqr_thresholds,
        "A recall": a_recall,
        "B recall": b_recall,
        "overall accuracy": overall,
    }
)

fig, ax = plt.subplots()
sns.lineplot(data=dataframe, x="iqr", y="A recall", label="A recall", ax=ax)
sns.lineplot(data=dataframe, x="iqr", y="B recall", label="B recall", ax=ax)
sns.lineplot(data=dataframe, x="iqr", y="overall accuracy", label="overall", ax=ax)

sns.scatterplot(data=dataframe, x="iqr", y="A recall", ax=ax)
sns.scatterplot(data=dataframe, x="iqr", y="B recall", ax=ax)
sns.scatterplot(data=dataframe, x="iqr", y="overall accuracy", ax=ax)

ax.invert_xaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color("#808080")
ax.spines["left"].set_color("#808080")
alpha = 0.3

ax.grid(alpha=alpha)
ax.set_xlabel("iqr threshold")
ax.set_ylabel("recall")
ax.set_title("event-wise accuracy and iqr threshold")

plt.show()

# iqr threshold, votes, and percent coverage event wise
