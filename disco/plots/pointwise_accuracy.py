import os
from glob import glob

import matplotlib.pyplot as plt

import disco.cfg as cfg

plt.style.use("ggplot")

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import disco.figure_utils as fig_util
import disco.util.inference_utils as util
from disco.accuracy_metrics import load_accuracy_metric_data


def compute_pointwise_accuracy(
    predictions,
    iqr,
    label_csv,
    sample_rate,
    hop_length,
):
    # load the labels
    label_df = pd.read_csv(label_csv).drop_duplicates()
    # create an array that's all background. The labels must annotate the whole file.
    label_array = np.zeros((iqr.shape[1])) + cfg.name_to_class_code["BACKGROUND"]

    # paint labels onto the label array based on the regions labeled in the .csv
    for _, row in label_df.iterrows():
        begin = row["Begin Time (s)"]
        end = row["End Time (s)"]
        label = row["Sound_Type"]
        begin_idx = util.convert_time_to_spect_index(begin, hop_length, sample_rate)
        end_idx = util.convert_time_to_spect_index(end, hop_length, sample_rate)
        label_array[begin_idx:end_idx] = cfg.name_to_class_code[label]

    # now compute confusion matrices
    conf_matrices = []
    # iterate over the iqr thresholds backwards (so largest IQR threshold comes first).
    # set any predictions with iqr >= threshold = background
    for threshold in fig_util.iqr_thresholds:
        predictions[np.max(iqr, axis=0) >= threshold] = cfg.name_to_class_code[
            "BACKGROUND"
        ]
        conf_matrices.append(
            confusion_matrix(y_true=label_array, y_pred=predictions, labels=[0, 1, 2])
        )
    return conf_matrices, fig_util.iqr_thresholds


if __name__ == "__main__":

    ensemble_to_cmat = {}

    for ensemble_directory in glob(fig_util.root + "/*ensemble*"):
        conf_matrix = None
        for test_directory, label_file in zip(
            fig_util.test_directories, fig_util.label_files
        ):
            data = load_accuracy_metric_data(
                os.path.join(fig_util.root, ensemble_directory, test_directory)
            )
            cmats, iqrs = compute_pointwise_accuracy(
                data["medians"],
                data["iqr"],
                label_csv=os.path.join(fig_util.root, label_file),
                sample_rate=48000,
                hop_length=200,
            )
            if conf_matrix is None:
                conf_matrix = cmats
            else:
                for i, cmat in enumerate(cmats):
                    conf_matrix[i] += cmat

        ensemble_to_cmat[os.path.basename(ensemble_directory)] = conf_matrix


fig, ax = plt.subplots(ncols=2, sharey=True, sharex=True, figsize=(12.2, 5))

ensemble_name_map = {
    "ensemble_2_random_init": "2 member, random init.",
    "ensemble_2_bootstrap": "2 member, bootstrap",
    "ensemble_10_random_init": "10 member, random init.",
    "ensemble_10_bootstrap": "10 member, bootstrap",
    "ensemble_30_random_init": "30 member, random init.",
    "ensemble_30_bootstrap": "30 member, bootstrap",
}
sorted_keys = [
    "ensemble_2_random_init",
    "ensemble_2_bootstrap",
    "ensemble_10_random_init",
    "ensemble_10_bootstrap",
    "ensemble_30_random_init",
    "ensemble_30_bootstrap",
]


for sorted_key in sorted_keys:
    ensemble_type = sorted_key
    confusion_matrices = ensemble_to_cmat[ensemble_type]
    # if ensemble_type != "ensemble_10_random_init":
    #     continue

    recalls = [
        np.diag((c / np.sum(c, axis=1, keepdims=True))) for c in confusion_matrices
    ]
    precisions = [
        np.diag((c / np.sum(c, axis=0, keepdims=True))) for c in confusion_matrices
    ]

    # A chirp precision and recall
    (line,) = ax[0].plot(
        [r[0] for r in recalls],
        [r[0] for r in precisions],
        "o-",
        label=ensemble_name_map[ensemble_type],
        markersize=5,
    )

    # B chirp precision and recall
    (line,) = ax[1].plot(
        [r[1] for r in recalls],
        [r[1] for r in precisions],
        "o-",
        label=ensemble_name_map[ensemble_type],
        markersize=5,
    )


ax[0].set_title("A chirp")
ax[1].set_title("B chirp")

leg = ax[1].legend(
    loc="best",
    fancybox=True,
    framealpha=0.5,
    frameon=False,
    fontsize=16,
    facecolor="white",
)

ax[0].set_ylabel("precision", fontsize=16, color="black")


ax[0].set_xlim(0, 1.01)
ax[1].set_xlim(0, 1.01)

subplot = fig.add_subplot(111, frameon=False)
plt.tick_params(
    labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
)
plt.xlabel("recall", color="black", fontsize=16)
subplot.spines["top"].set_visible(False)
subplot.spines["right"].set_visible(False)
subplot.spines["bottom"].set_visible(False)
subplot.spines["left"].set_visible(False)
subplot.set_facecolor("none")
subplot.grid(alpha=0.0, color="#808080")

for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.spines["bottom"].set_color("#808080")
    a.spines["left"].set_color("#808080")
    a.set_facecolor("none")
    a.grid(alpha=0.1, color="#808080")

plt.savefig(
    f"{os.path.join(os.environ['HOME'], 'pointwise_accuracy.pdf')}",
    format="pdf",
    bbox_inches="tight",
    dpi=600,
)
plt.show()
