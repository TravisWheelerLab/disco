import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import disco.cfg as cfg
import disco.util.inference_utils as util
from disco.accuracy_metrics import eventwise_metrics, load_accuracy_metric_data

labels = glob(
    "/Users/wheelerlab/beetles_testing/unannotated_files_12_12_2022/audio_2020/*csv"
)
# inject bits of background in between each labeled chirp.
hop_length = 200
# this is special - it's a one-off analysis because I've actually labeled the entire file
cmat = None


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

    # root = "/Users/wheelerlab/beetles_testing/unannotated_files_12_12_2022/audio_2020/"
    root = "/xdisk/twheeler/colligan/ground_truth/"
    # compare the different ensembles.
    test1 = "180101_0133S12-viz"
    test2 = "180101_0183S34D06-viz"
    test3 = "trial40_M57_F29_070220-viz"
    test_directories = [test1, test2, test3]

    label_files = [
        "180101_0133S12.csv",
        "180101_0183S34D06.csv",
        "trial40_M57_F29_070220.csv",
    ]

    ensemble_to_cmat = {}

    for ensemble_directory in glob(root + "/snr_ablation/*ensemble*"):
        conf_matrix = None
        for test_directory, label_file in zip(test_directories, label_files):
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

fig, ax = plt.subplots(ncols=2, sharey=True, sharex=True, figsize=(13, 10))

ensemble_name_map = {
    "ensemble_10_random_init": "10 member, random init.",
    "ensemble_2_random_init": "2 member, random init.",
    "ensemble_30_random_init": "30 member, random init.",
    "ensemble_2_bootstrap": "2 member, bootstrap",
    "ensemble_10_bootstrap": "10 member, bootstrap",
    "ensemble_30_bootstrap": "30 member, bootstrap",
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
                snr = 500

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
    (line,) = ax[0].plot(
        snrs,
        [r[0] for r in recalls],
        "o-",
        label=ensemble_type,
        markersize=8,
    )
    # B chirp
    ax[1].plot(
        snrs, [r[1] for r in recalls], "o-", color=line.get_color(), markersize=8
    )

ax[0].legend()
ax[0].set_ylabel("recall")

ax[0].invert_xaxis()
ax[1].invert_xaxis()

ax[0].semilogx()
ax[1].semilogx()

ax[0].set_title("A chirp")
ax[1].set_title("B chirp")

labels = ax[0].get_xticks()
labels[0] = "more noise"
labels[-1] = "less noise"

ax[0].set_xticklabels(labels)
ax[1].set_xticklabels(labels)

fig.text(y=0.05, x=0.5, s="sn ratio", ha="center")

for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.spines["bottom"].set_color("#808080")
    a.spines["left"].set_color("#808080")

plt.savefig("testing.png", bbox_inches="tight")
