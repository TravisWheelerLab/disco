import os
from collections import defaultdict
from copy import deepcopy
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import disco.cfg as cfg
import disco.util.inference_utils as util
from disco.accuracy_metrics import load_accuracy_metric_data


def fill_gaps_in_as_and_bs(arr, min_length=10):
    (transitions,) = np.where(np.diff(arr) != 0)
    # add the 0 boundary on so we capture the first label
    # also add the end so we capture the last label
    boundaries = np.concatenate(([0], transitions, [arr.shape[0]]))
    i = 1
    smoothed_arr = deepcopy(arr)
    for i in range(1, len(boundaries) - 2):

        prev_slice = arr[boundaries[i - 1] + 1 : boundaries[i + 1] + 1]
        current_slice = arr[boundaries[i] + 1 : boundaries[i + 1] + 1]
        next_slice = arr[boundaries[i + 1] + 1 : boundaries[i + 2] + 2]
        # if the previous and current match and the middle slice
        # is not the same class
        if (prev_slice[0] == next_slice[0]) and (current_slice[0] != prev_slice[0]):
            # we're gap filling background bits inside of As and Bs
            # A X A X A
            if prev_slice[0] in (
                cfg.name_to_class_code["A"],
                cfg.name_to_class_code["B"],
            ):
                if len(current_slice) <= min_length:
                    # if it's a little gap, fill it with the adjacent classes
                    smoothed_arr[
                        boundaries[i - 1] : boundaries[i + 1] + 1
                    ] = prev_slice[0]

    return smoothed_arr


def fill_gaps_in_background(arr, min_length=10):
    (transitions,) = np.where(np.diff(arr) != 0)
    # add the 0 boundary on so we capture the first label
    # also add the end so we capture the last label
    boundaries = np.concatenate(([0], transitions, [arr.shape[0]]))
    i = 1
    smoothed_arr = deepcopy(arr)
    for i in range(1, len(boundaries) - 2):

        prev_slice = arr[boundaries[i - 1] + 1 : boundaries[i + 1] + 1]
        current_slice = arr[boundaries[i] + 1 : boundaries[i + 1] + 1]
        next_slice = arr[boundaries[i + 1] + 1 : boundaries[i + 2] + 2]
        # if the previous and current match and the middle slice
        # is not the same class
        if (prev_slice[0] == next_slice[0]) and (current_slice[0] != prev_slice[0]):
            if prev_slice[0] == cfg.name_to_class_code["BACKGROUND"]:
                if len(current_slice) <= min_length:
                    # if it's a little gap, fill it with the adjacent classes
                    smoothed_arr[
                        boundaries[i - 1] : boundaries[i + 1] + 1
                    ] = prev_slice[0]

    return smoothed_arr


def load_and_compute(
    ensemble_directory,
    test_directory,
    label_file,
    iqr_threshold,
    root,
    single_model=False,
    fill_gaps=False,
):

    data_dict = load_accuracy_metric_data(
        os.path.join(ensemble_directory, test_directory)
    )

    # paint labels onto a 1-D array
    label_array = create_label_array(
        label_csv=os.path.join(root, label_file), spectrogram=data_dict["spectrogram"]
    )

    predictions = data_dict["medians"]

    if fill_gaps:
        smoothed = fill_gaps_in_as_and_bs(predictions, min_length=15)
        predictions = fill_gaps_in_background(smoothed, min_length=15)

    if single_model:
        iqr = data_dict["raw_preds"]
        _iqrs = []
        # select the "iqr" based on the prediction -
        for kk, pred in enumerate(predictions):
            _iqrs.append(iqr[pred, kk])
        iqr = np.asarray(_iqrs)
        predictions[iqr <= iqr_threshold] = cfg.name_to_class_code["BACKGROUND"]
    else:
        iqr = np.max(data_dict["iqr"], axis=0)
        predictions[iqr >= iqr_threshold] = cfg.name_to_class_code["BACKGROUND"]

    # compute total labeled chirps
    label_to_true = count_contiguous_regions(label_array)
    # compute total predictions
    label_to_predicted = count_contiguous_regions(predictions)
    # compute hits and misses
    label_to_hit_and_miss = count_overlapping_predictions(label_array, predictions)

    return label_to_true, label_to_predicted, label_to_hit_and_miss


def compute_recall(label_to_hit_and_miss, class_name):
    num = 0
    denom = 0

    for entry in label_to_hit_and_miss:
        num += entry[class_name]["hit"]
        denom += entry[class_name]["hit"] + entry[class_name]["miss"]

    return num / denom


def compute_precision(label_to_hit_and_miss, label_to_predicted, class_name):
    num = 0
    denom = 0

    for hits_and_misses, pred_counts in zip(label_to_hit_and_miss, label_to_predicted):
        num += hits_and_misses[class_name]["hit"]
        denom += pred_counts[class_name]

    return num / denom


def count_overlapping_predictions(labels, predictions, min_length=10):
    (transitions,) = np.where(np.diff(labels) != 0)
    # add the 0 boundary on so we capture the first label
    # also add the end so we capture the last label
    boundaries = np.concatenate(([0], transitions, [labels.shape[0]]))
    label_to_hits = defaultdict(lambda: defaultdict(int))

    for i in range(len(boundaries) - 1):
        # np.diff returns locations adjacent to transitions
        # this goes from 0 -> len()-2
        contig = labels[boundaries[i] + 1 : boundaries[i + 1]]
        pred_contig = predictions[boundaries[i] + 1 : boundaries[i + 1]]
        # how many unique labels are within this little contiguous prediction?
        (pred_transitions,) = np.where(np.diff(pred_contig) != 0)
        # add the 0 boundary on so we capture the first label
        # also add the end so we capture the last label
        # if there aren't any transitions within this predicted slice,
        if len(pred_transitions) == 0:
            # create a new boundary array that's just the beginning
            # and end of the prediction slice
            pred_boundaries = np.concatenate(([0], [pred_contig.shape[0]]))
        else:
            # otherwise, add the beginning and end onto the prediction slice
            pred_boundaries = np.concatenate(
                ([0], pred_transitions, [pred_contig.shape[0]])
            )

        if len(contig) == 0:
            continue

        y_true = contig[0]

        hit = False

        for j in range(len(pred_boundaries) - 1):
            sub_pred_contig = pred_contig[
                pred_boundaries[j] + 1 : pred_boundaries[j + 1]
            ]
            # if it's above the min length and it's the same class
            if len(sub_pred_contig) >= min_length and y_true == sub_pred_contig[0]:
                # consider it a ground-truth hit, log it, and break out of this
                # ground-truth section
                hit = True
                break
            # otherwise, don't do anything, and continue: do we have any other little
            # labeled regions in this chirp that will save us?
        if hit:
            label_to_hits[cfg.class_code_to_name[y_true]]["hit"] += 1
        else:
            label_to_hits[cfg.class_code_to_name[y_true]]["miss"] += 1

    return label_to_hits


def count_contiguous_regions(labels, min_length=10):
    """
    Count all contiguous classified regions in labels above min_length.
    """
    (transitions,) = np.where(np.diff(labels) != 0)
    # add the 0 boundary on so we capture the first label
    # also add the end so we capture the last label
    boundaries = np.concatenate(([0], transitions, [labels.shape[0]]))
    label_to_count = defaultdict(int)

    for i in range(len(boundaries) - 1):
        # np.diff returns locations adjacent to transitions
        # this goes from 0 -> len()-2
        contig = labels[boundaries[i] + 1 : boundaries[i + 1]]

        if len(contig) == 0:
            continue

        assert len(np.unique(contig)) == 1

        if len(contig) >= min_length:
            label_to_count[cfg.class_code_to_name[contig[0]]] += 1

    return label_to_count


def create_label_array(
    label_csv, spectrogram, sample_rate=48000, hop_length=200, min_length=15
):
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
        if label == "BACKGROUND":
            # it's the default, so forget about it.
            continue

        begin_idx = util.convert_time_to_spect_index(begin, hop_length, sample_rate)
        end_idx = util.convert_time_to_spect_index(end, hop_length, sample_rate)

        if (end_idx - begin_idx) <= min_length:
            continue

        label_array[begin_idx:end_idx] = cfg.name_to_class_code[label]

    (transitions,) = np.where(np.diff(label_array) != 0)
    # add the 0 boundary on so we capture the first label
    # also add the end so we capture the last label

    return label_array


if __name__ == "__main__":

    root = "/Users/wheelerlab/beetles_testing/unannotated_files_12_12_2022/audio_2020/"
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

    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    ensemble_directories = glob(root + "*ensemble*")
    iqr_thresholds = np.logspace(-2, 0, num=50)[::-1]

    recalls = defaultdict(lambda: defaultdict(list))
    precisions = defaultdict(lambda: defaultdict(list))

    for ensemble_directory in ensemble_directories:
        print(ensemble_directory)

        for iqr_threshold in iqr_thresholds:
            trues = []
            preds = []
            hits_and_misses = []

            for test_directory, label_file in zip(test_directories, label_files):
                # load the data
                (
                    label_to_true,
                    label_to_predicted,
                    label_to_hit_and_miss,
                ) = load_and_compute(
                    ensemble_directory=ensemble_directory,
                    test_directory=test_directory,
                    label_file=label_file,
                    iqr_threshold=iqr_threshold,
                    root=root,
                )
                trues.append(label_to_true)
                preds.append(label_to_predicted)
                hits_and_misses.append(label_to_hit_and_miss)

            # now, we can get precision and recall
            # recall is easy: use the hit and miss dictionary
            a_recall = compute_recall(hits_and_misses, class_name="A")
            b_recall = compute_recall(hits_and_misses, class_name="B")

            a_precision = compute_precision(hits_and_misses, preds, class_name="A")

            b_precision = compute_precision(hits_and_misses, preds, class_name="B")

            recalls[os.path.basename(ensemble_directory)]["A"].append(a_recall)
            recalls[os.path.basename(ensemble_directory)]["B"].append(b_recall)

            precisions[os.path.basename(ensemble_directory)]["A"].append(a_precision)
            precisions[os.path.basename(ensemble_directory)]["B"].append(b_precision)

        # now for precision: of everything we predicted as a class, how many were actually correct?
        # well, we can get everything we predicted as a class by looking at
        # label_to_predicted. We know that label_to_hit_and_miss[class]["hit"] is how many we
        # got correct.
        # This is going to underestimate precision because
        # two B chirps can be separated by 2 pixels but a-priori we have no clue that that is
        # actually one chirp. Only after the application of some gap-filling algorithms can
        # we say that the prediction is one chirp. A boxcar mean with a small window (like 10?)
        # can fix this, maybe. Might be easier to just say: if there are two sounds of the same
        # class separated by just a little bit, send the little bit to the same class.

    for ensemble_type in recalls:
        r = recalls[ensemble_type]
        p = precisions[ensemble_type]

        (line,) = ax[0].plot(r["A"], p["A"], "-")
        ax[1].plot(r["B"], p["B"], "-", color=line.get_color(), label=ensemble_type)

    ax[0].set_xlim(0, 1)
    ax[1].set_xlim(0, 1)
    ax[1].legend()

    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1)

    ax[0].set_title("A chirp")
    ax[1].set_title("B chirp")

    ax[0].set_xlabel("recall")
    ax[1].set_xlabel("recall")

    ax[0].set_ylabel("precision")
    ax[1].set_ylabel("precision")

    plt.show()
