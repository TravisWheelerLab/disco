import seaborn as sns

sns.set_palette(sns.color_palette("crest"))

import matplotlib.pyplot as plt
import pandas as pd

from disco.plots.plot import Plotter, outcome_variables

"""
This script is intended to plot event-wise accuracy at a single iqr threshold over a range of different
signal-to-noise ratios.
It uses a 10-member ensemble, random initialization, and ablates over signal-to-noise ratio.

"""
# use a signal-to-noise ratio of 0, since this indicates unnoised data.
snrs = [10, 15, 20, 25, 30, 35, 40, 80, 160, 320, 0]
overall = []
a = []
b = []

for snr in snrs:
    # this might be a little annoying.
    # it's probably going to be easier to just not use the plotter object.
    snr_file = f"disco/resources/disco_accuracy_csvs/snr_{snr}_ensemble_10_random_init/event_wise_10_random_init.csv"
    df = pd.read_csv(snr_file, index_col=0)
    # iqr threshold, votes, and percent coverage event wise
    pct_30 = df["1/0/30"]
    overall_acc = float(pct_30["accuracy"])
    a_acc = float(pct_30["recall, A chirp"])
    b_acc = float(pct_30["recall, B chirp"])
    overall.append(overall_acc)
    a.append(a_acc)
    b.append(b_acc)

snrs[-1] = 350

dataframe = pd.DataFrame.from_dict(
    {"snr": snrs, "overall_accuracy": overall, "A": a, "B": b}
)
fig, ax = plt.subplots()

sns.lineplot(
    data=dataframe, ax=ax, x="snr", y="overall_accuracy", label="overall accuracy"
)
sns.lineplot(data=dataframe, ax=ax, x="snr", y="A", label="A chirp")
sns.lineplot(data=dataframe, ax=ax, x="snr", y="B", label="B chirp")

sns.scatterplot(data=dataframe, ax=ax, x="snr", y="overall_accuracy")
sns.scatterplot(data=dataframe, ax=ax, x="snr", y="A")
sns.scatterplot(data=dataframe, ax=ax, x="snr", y="B")

ax.invert_xaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color("#808080")
ax.spines["left"].set_color("#808080")
alpha = 0.3

ax.grid(alpha=alpha)
ax.set_xlabel("signal-to-noise ratio")
ax.set_ylabel("accuracy")
ax.set_title("accuracy and signal-to-noise")
ax.semilogx()

plt.show()
