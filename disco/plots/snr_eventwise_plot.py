import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import disco.cfg as cfg
import disco.figure_utils as fig_util
import disco.util.inference_utils as util
from disco.accuracy_metrics import eventwise_metrics, load_accuracy_metric_data


def compute_eventwise_accuracy(
    predictions,
    spectrogram,
    label_csv,
    sample_rate,
    hop_length,
):
    # load the labels
    label_df = pd.read_csv(label_csv).drop_duplicates()
    # create an array that's all background. The labels must annotate the whole file.
    label_array = (
        np.zeros((spectrogram.shape[1])) + cfg.name_to_class_code["BACKGROUND"]
    )

    # paint labels onto the label array based on the regions labeled in the .csv
    for _, row in label_df.iterrows():
        begin = row["Begin Time (s)"]
        end = row["End Time (s)"]
        label = row["Sound_Type"]
        begin_idx = util.convert_time_to_spect_index(begin, hop_length, sample_rate)
        end_idx = util.convert_time_to_spect_index(end, hop_length, sample_rate)
        label_array[begin_idx:end_idx] = cfg.name_to_class_code[label]

    # now chunk the array
    (transitions,) = np.where(np.diff(label_array) != 0)
    ground_truth = []
    corresponding_predictions = []
    cmat = None

    for i in range(len(transitions) - 1):
        if transitions[i + 1] - transitions[i] < 5:
            continue

        prediction_slice = predictions[transitions[i] + 1 : transitions[i + 1]]
        ground_truth_slice = label_array[transitions[i] + 1 : transitions[i + 1]]
        assert len(np.unique(ground_truth_slice)) == 1
        ground_truth.append(ground_truth_slice)
        corresponding_predictions.append(prediction_slice)
        # now VISUALIZE!
        spectrogram_slice = spectrogram[:, transitions[i] + 1 : transitions[i + 1]]

    data_dict = {
        "medians": corresponding_predictions,
        "ground_truth": ground_truth,
        "spectrograms": [],
    }

    # returns an array of confusion matrices at different coverage percentages
    _, _, cmats, cov_pcts = eventwise_metrics(data_dict, cov_pcts=[0.5])

    if cmat is None:
        cmat = cmats
    else:
        for i, c in enumerate(cmats):
            cmat[i] += c

    return cmats, cov_pcts


if __name__ == "__main__":

    root = "/xdisk/twheeler/colligan/ground_truth/"
    # compare the different ensembles.

    ensemble_to_cmat = {}

    for ensemble_directory in glob(root + "/snr_ablation/*ensemble*"):
        conf_matrix = None
        for test_directory, label_file in zip(
            fig_util.test_directories, fig_util.label_files
        ):
            data = load_accuracy_metric_data(
                os.path.join(root, ensemble_directory, test_directory)
            )

            conf_matrices, cov_pct = compute_eventwise_accuracy(
                predictions=data["medians"],
                spectrogram=data["spectrogram"],
                label_csv=os.path.join(root, label_file),
                sample_rate=48000,
                hop_length=200,
            )
            if conf_matrix is None:
                conf_matrix = conf_matrices
            else:
                for i, cmat in enumerate(conf_matrices):
                    conf_matrix[i] += cmat

        ensemble_to_cmat[os.path.basename(ensemble_directory)] = conf_matrix

fig, ax = plt.subplots(figsize=(8.2, 5))

ensemble_name_map = {
    "ensemble_10_random_init": "10 member, random init.",
    # "ensemble_2_random_init": "2 member, random init.",
    # "ensemble_30_random_init": "30 member, random init.",
    # "ensemble_2_bootstrap": "2 member, bootstrap",
    # "ensemble_10_bootstrap": "10 member, bootstrap",
    # "ensemble_30_bootstrap": "30 member, bootstrap",
}

snr = []
accuracy = []

ensemble_to_snr = {}

for name in ensemble_name_map:
    performances = []
    snrs = []
    for snr_name in ensemble_to_cmat.keys():
        if name in snr_name:
            # only will work with one cov. percent
            (cmat,) = ensemble_to_cmat[snr_name]
            performances.append(cmat)
            snr = int(snr_name[: snr_name.find("_")])

            if snr == 0:
                # 0 is shorthand for "no noise"
                snr = 300

            snrs.append(snr)

    ensemble_to_snr[name] = (snrs, performances)

# plot SNR and recall

for ensemble_type, arrs in ensemble_to_snr.items():
    snrs, confusion_matrices = arrs
    sorted_idx = np.argsort(snrs)
    snrs = np.asarray(snrs)[sorted_idx]
    confusion_matrices = np.asarray(confusion_matrices)[sorted_idx]

    recalls = [
        np.diag((c / np.sum(c, axis=1, keepdims=True))) for c in confusion_matrices
    ]
    # A chirp
    (line,) = ax.plot(
        snrs[:-1],
        [r[0] for r in recalls][:-1],
        "o-",
        c="b",
        label="A chirp",
    )
    # B chirp
    ax.plot(snrs[:-1], [r[1] for r in recalls][:-1], "o-", color="r", label="B chirp")

leg = ax.legend(
    loc="best", framealpha=0.5, frameon=False, facecolor="white", fontsize=16
)

ax.set_ylabel("recall", fontsize=16)

fig.text(y=0.030, x=0.5, s="sn ratio", ha="center", fontsize=16)

ax.grid(alpha=0.1, color="#808080")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color("#808080")
ax.spines["left"].set_color("#808080")
ax.invert_xaxis()

plt.suptitle("A, B recall as a function of SN ratio", fontsize=14)

plt.savefig(
    f"{os.path.join(os.environ['HOME'], 'snr_eventwise.pdf')}",
    format="pdf",
    bbox_inches="tight",
    dpi=600,
)
