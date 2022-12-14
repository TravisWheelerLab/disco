import os
from glob import glob

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import disco.cfg as cfg
import disco.util.inference_utils as util
from disco.accuracy_metrics import load_accuracy_metric_data

labels = glob(
    "/Users/wheelerlab/beetles_testing/unannotated_files_12_12_2022/audio_2020/*csv"
)
# inject bits of background in between each labeled chirp.
hop_length = 200
# this is special - it's a one-off analysis because I've actually labeled the entire file
cmat = None


def compute_pointwise_accuracy(
    predictions,
    hmm_predictions,
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
    hmm_conf_matrices = []
    iqr_thresholds = np.logspace(-1, 0, num=10)[::-1]
    # iterate over the iqr thresholds backwards (so largest IQR threshold comes first).
    # set any predictions with iqr >= threshold = background
    for threshold in iqr_thresholds:
        predictions[np.max(iqr, axis=0) >= threshold] = cfg.name_to_class_code[
            "BACKGROUND"
        ]
        hmm_predictions[np.max(iqr, axis=0) >= threshold] = cfg.name_to_class_code[
            "BACKGROUND"
        ]

        conf_matrices.append(
            confusion_matrix(y_true=label_array, y_pred=predictions, labels=[0, 1, 2])
        )
        hmm_conf_matrices.append(
            confusion_matrix(
                y_true=label_array, y_pred=hmm_predictions, labels=[0, 1, 2]
            )
        )

    return conf_matrices, hmm_conf_matrices, iqr_thresholds


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
            cmats, hmm_cmats, iqrs = compute_pointwise_accuracy(
                data["medians"],
                util.smooth_predictions_with_hmm(
                    data["medians"],
                    cfg.hmm_transition_probabilities,
                    cfg.hmm_emission_probabilities,
                    cfg.hmm_start_probabilities,
                ),
                data["iqr"],
                label_csv=os.path.join(root, label_file),
                sample_rate=48000,
                hop_length=200,
            )
            if conf_matrix is None:
                conf_matrix = cmats
                hmm_conf_matrix = hmm_cmats
            else:
                for i, cmat in enumerate(cmats):
                    conf_matrix[i] += cmat

                for i, cmat in enumerate(hmm_cmats):
                    hmm_conf_matrix[i] += cmat

        ensemble_to_cmat[os.path.basename(ensemble_directory)] = conf_matrix
        ensemble_to_hmm_cmat[os.path.basename(ensemble_directory)] = hmm_conf_matrix

    fig, ax = plt.subplots(ncols=2, sharey=True, sharex=True)

    ensemble_name_map = {
        "ensemble_10_random_init": "10",
        "ensemble_2_random_init": "2",
        "ensemble_30_random_init": "30",
        "ensemble_2_bootstrap": "2 member, bootstrap",
        "ensemble_10_bootstrap": "10 member, bootstrap",
        "ensemble_30_bootstrap": "30 member, bootstrap",
    }

    for ensemble_type, confusion_matrices in ensemble_to_cmat.items():

        if "random_init" not in ensemble_type:
            continue

        recalls = [
            np.diag((c / np.sum(c, axis=1, keepdims=True))) for c in confusion_matrices
        ]
        precisions = [
            np.diag((c / np.sum(c, axis=0, keepdims=True))) for c in confusion_matrices
        ]

        hmm_confusion_matrices = ensemble_to_hmm_cmat[ensemble_type]

        hmm_recalls = [
            np.diag((c / np.sum(c, axis=1, keepdims=True)))
            for c in hmm_confusion_matrices
        ]
        hmm_precisions = [
            np.diag((c / np.sum(c, axis=0, keepdims=True)))
            for c in hmm_confusion_matrices
        ]

        # A chirp recall, pre and post hmm
        (line1,) = ax[0].plot(
            iqrs,
            [r[0] for r in recalls],
            "o-",
            label=ensemble_name_map[ensemble_type] + " recall",
            markersize=5,
        )
        ax[0].plot(
            iqrs,
            [r[0] for r in hmm_recalls],
            "*-",
            color=line1.get_color(),
            markersize=7,
        )

        # A chirp precision, pre and post hmm
        (line2,) = ax[0].plot(
            iqrs,
            [r[0] for r in precisions],
            "o-",
            label=ensemble_name_map[ensemble_type] + " precision",
            markersize=5,
        )
        ax[0].plot(
            iqrs,
            [r[0] for r in hmm_precisions],
            "*-",
            color=line2.get_color(),
            markersize=7,
        )

        # B chirp recall, pre and post hmm
        (line,) = ax[1].plot(
            iqrs,
            [r[1] for r in recalls],
            "o-",
            color=line1.get_color(),
            label=ensemble_name_map[ensemble_type] + " recall",
            markersize=5,
        )
        ax[1].plot(
            iqrs,
            [r[1] for r in hmm_recalls],
            "*-",
            color=line1.get_color(),
            markersize=7,
        )

        # B chirp precision, pre and post hmm
        (line,) = ax[1].plot(
            iqrs,
            [r[1] for r in precisions],
            "o-",
            color=line2.get_color(),
            label=ensemble_name_map[ensemble_type] + " precision",
            markersize=5,
        )
        ax[1].plot(
            iqrs,
            [r[1] for r in hmm_precisions],
            "*-",
            color=line2.get_color(),
            markersize=7,
        )

    ax[0].semilogx()
    ax[1].semilogx()

    ax[0].set_title("A chirp precision and recall, pre and post hmm")
    ax[1].set_title("B chirp precision and recall, pre and post hmm")

    ax[0].legend()
    ax[0].set_ylabel("precision/recall")

    fig.text(y=0.05, x=0.5, s="iqr threshold", ha="center")

    black_star = mlines.Line2D(
        [],
        [],
        color="black",
        marker="*",
        linestyle="None",
        markersize=10,
        label="post-hmm",
    )

    black_dot = mlines.Line2D(
        [],
        [],
        color="black",
        marker="o",
        linestyle="None",
        markersize=6,
        label="pre-hmm",
    )
    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.spines["bottom"].set_color("#808080")
        a.spines["left"].set_color("#808080")

    ax[1].legend(handles=[black_star, black_dot])

    plt.show()
