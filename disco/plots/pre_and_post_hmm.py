import seaborn as sns

sns.set_palette(sns.color_palette("crest"))

import matplotlib.pyplot as plt
import pandas as pd

"""
This script is intended to compare results pre-and-post hmm, for both event-wise and
pointwise accuracy computation.
Both use the 10 member randomly initialized ensemble.
"""
iqrs = [1, 0.4, 0.2, 0.1, 0.05]
coverages = [30, 40, 50, 70, 80, 90, 95]


def load_data_eventwise(root, class_name):
    results = []

    for method in ["random_init"]:
        for ensemble_size in [10]:
            results_file = f"{root}/event_wise_{ensemble_size}_{method}.csv"
            results_df = pd.read_csv(results_file, index_col=0)
            for cover in coverages:
                recall = float(
                    results_df[f"1/0/{cover}"][f"recall, {class_name} chirp"]
                )
                results_dict = {
                    "members": ensemble_size,
                    "method": method,
                    "coverage": cover,
                    "recall": recall,
                    "class": class_name,
                }
                results.append(results_dict)

    results = pd.DataFrame.from_dict(results)
    return results


def load_data_pointwise(root, class_name):
    results = []

    for method in ["random_init"]:
        for ensemble_size in [10]:
            results_file = f"{root}/point_wise_{ensemble_size}_{method}.csv"
            results_df = pd.read_csv(results_file, index_col=0)
            for iqr in iqrs:
                recall = float(results_df[f"{iqr}/0"][f"recall, {class_name} chirp"])
                precision = float(
                    results_df[f"{iqr}/0"][f"precision, {class_name} chirp"]
                )
                f1 = (2 * recall * precision) / (recall + precision)
                results_dict = {
                    "members": ensemble_size,
                    "method": method,
                    "iqr": iqr,
                    "f1": f1,
                    "class": class_name,
                }
                results.append(results_dict)

    results = pd.DataFrame.from_dict(results)
    return results


pre_hmm_root = f"disco/resources/disco_accuracy_csvs/snr_0/"
post_hmm_root = f"disco/resources/disco_hmm_accuracy_csvs/snr_0/"

pre_hmm_a = load_data_pointwise(pre_hmm_root, "A")
pre_hmm_b = load_data_pointwise(pre_hmm_root, "B")

post_hmm_a = load_data_pointwise(post_hmm_root, "A")
post_hmm_b = load_data_pointwise(post_hmm_root, "B")

post_hmm_a["hmm applied"] = True
post_hmm_b["hmm applied"] = True

pre_hmm_a["hmm applied"] = False
pre_hmm_b["hmm applied"] = False
# merge the df
both = pd.concat((pre_hmm_a, post_hmm_b, pre_hmm_b, post_hmm_a))

fig, ax = plt.subplots(ncols=2, sharey=True)
ax[1].set_ylabel("recall")

sns.lineplot(
    data=both,
    x="iqr",
    y="f1",
    hue="hmm applied",
    style="class",
    palette="crest",
    legend=None,
    ax=ax[0],
)

sns.scatterplot(
    data=both,
    x="iqr",
    y="f1",
    style="class",
    hue="hmm applied",
    palette="crest",
    ax=ax[0],
)

alpha = 0.3

pre_hmm_a = load_data_eventwise(pre_hmm_root, "A")
pre_hmm_b = load_data_eventwise(pre_hmm_root, "B")

post_hmm_a = load_data_eventwise(post_hmm_root, "A")
post_hmm_b = load_data_eventwise(post_hmm_root, "B")

post_hmm_a["hmm applied"] = True
post_hmm_b["hmm applied"] = True

pre_hmm_a["hmm applied"] = False
pre_hmm_b["hmm applied"] = False
# merge the df
both = pd.concat((pre_hmm_a, post_hmm_b, pre_hmm_b, post_hmm_a))


sns.lineplot(
    data=both,
    x="coverage",
    y="recall",
    hue="hmm applied",
    style="class",
    palette="crest",
    legend=None,
    ax=ax[1],
)

sns.scatterplot(
    data=both,
    x="coverage",
    y="recall",
    style="class",
    hue="hmm applied",
    palette="crest",
    ax=ax[1],
    legend=None,
)

for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.spines["bottom"].set_color("#808080")
    a.spines["left"].set_color("#808080")
    a.grid(alpha=alpha)

ax[0].invert_xaxis()

plt.suptitle("A and B chirp pointwise and eventwise metrics, pre and post-hmm")
plt.show()
