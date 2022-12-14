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
    hmm_predictions,
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
    corresponding_hmm_predictions = []
    cmat = None
    hmm_cmat = None

    for i in range(len(transitions) - 1):
        if transitions[i + 1] - transitions[i] < 5:
            continue

        prediction_slice = predictions[transitions[i] + 1 : transitions[i + 1]]
        hmm_prediction_slice = hmm_predictions[transitions[i] + 1 : transitions[i + 1]]
        ground_truth_slice = label_array[transitions[i] + 1 : transitions[i + 1]]
        assert len(np.unique(ground_truth_slice)) == 1
        ground_truth.append(ground_truth_slice)
        corresponding_predictions.append(prediction_slice)
        corresponding_hmm_predictions.append(hmm_prediction_slice)
        # now VISUALIZE!
        spectrogram_slice = spectrogram[:, transitions[i] + 1 : transitions[i + 1]]

    data_dict = {"medians": corresponding_predictions, "ground_truth": ground_truth}
    hmm_data_dict = {
        "medians": corresponding_hmm_predictions,
        "ground_truth": ground_truth,
    }

    # returns an array of confusion matrices at different coverage percentages
    _, _, cmats, cov_pcts = eventwise_metrics(data_dict)
    _, _, hmm_cmats, cov_pcts = eventwise_metrics(hmm_data_dict)

    if cmat is None:
        cmat = cmats
        hmm_cmat = hmm_cmats
    else:
        for i, c in enumerate(cmats):
            cmat[i] += c

        for i, c in enumerate(hmm_cmats):
            hmm_cmat[i] += c

    return cmats, hmm_cmats, cov_pcts


if __name__ == "__main__":

    root = "/Users/wheelerlab/beetles_testing/unannotated_files_12_12_2022/audio_2020"
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
    ensemble_to_hmm_cmat = {}

    for ensemble_directory in glob(root + "/*ensemble*"):
        conf_matrix = None
        hmm_conf_matrix = None
        for test_directory, label_file in zip(test_directories, label_files):
            data = load_accuracy_metric_data(
                os.path.join(root, ensemble_directory, test_directory)
            )

            conf_matrices, hmm_conf_matrices, cov_pct = compute_eventwise_accuracy(
                predictions=data["medians"],
                hmm_predictions=util.smooth_predictions_with_hmm(
                    data["medians"],
                    cfg.hmm_transition_probabilities,
                    cfg.hmm_emission_probabilities,
                    cfg.hmm_start_probabilities,
                ),
                spectrogram=data["spectrogram"],
                label_csv=os.path.join(root, label_file),
                sample_rate=48000,
                hop_length=200,
            )
            if conf_matrix is None:
                conf_matrix = conf_matrices
                hmm_conf_matrix = hmm_conf_matrices
            else:
                for i, cmat in enumerate(conf_matrices):
                    conf_matrix[i] += cmat
                for i, cmat in enumerate(hmm_conf_matrices):
                    hmm_conf_matrix[i] += cmat

        ensemble_to_cmat[os.path.basename(ensemble_directory)] = conf_matrix
        ensemble_to_hmm_cmat[os.path.basename(ensemble_directory)] = hmm_conf_matrix


fig, ax = plt.subplots(
    ncols=2, sharey=True, sharex=True, subplot_kw={"aspect": "equal"}
)

ensemble_name_map = {
    "ensemble_10_random_init": "10",
    "ensemble_2_random_init": "2",
    "ensemble_30_random_init": "30",
    "ensemble_2_bootstrap": "2 member, bootstrap",
    "ensemble_10_bootstrap": "10 member, bootstrap",
    "ensemble_30_bootstrap": "30 member, bootstrap",
}


for ensemble_type, confusion_matrices in ensemble_to_cmat.items():
    if "10" and "random" not in ensemble_type:
        continue

    hmm_confusion_matrices = ensemble_to_hmm_cmat[ensemble_type]
    recalls = [
        np.diag((c / np.sum(c, axis=1, keepdims=True))) for c in confusion_matrices
    ]
    hmm_recalls = [
        np.diag((c / np.sum(c, axis=1, keepdims=True))) for c in hmm_confusion_matrices
    ]

    (line,) = ax[0].plot(
        cov_pct,
        [r[0] for r in recalls],
        "*-",
        label=ensemble_name_map[ensemble_type],
        markersize=8,
    )
    ax[1].plot(
        cov_pct, [r[1] for r in recalls], "*-", color=line.get_color(), markersize=5
    )

    (line,) = ax[0].plot(
        cov_pct,
        [r[0] for r in hmm_recalls],
        "o-",
        label=ensemble_name_map[ensemble_type] + " post hmm",
        markersize=5,
    )

    ax[1].plot(
        cov_pct, [r[1] for r in hmm_recalls], "o-", color=line.get_color(), markersize=5
    )


ax[0].legend()
ax[0].set_ylabel("accuracy")

ax[0].set_title("A chirp")
ax[1].set_title("B chirp")

fig.text(y=0.1, x=0.5, s="percent event covered", ha="center")

for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.spines["bottom"].set_color("#808080")
    a.spines["left"].set_color("#808080")

plt.show()
