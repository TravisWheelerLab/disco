import os
from collections import defaultdict

import numpy as np
import pandas as pd

import disco_sound.cfg as cfg
import disco_sound.util.inference_utils as util
from disco_sound.plots.accuracy_metrics import load_accuracy_metric_data

# constants
iqr_thresholds = np.logspace(-2.77, np.log(0.999), num=50)[::-1]
softmax_thresholds = np.logspace(-2, np.log(0.9999), num=50)
test1 = "180101_0133S12-viz"
test2 = "180101_0183S34D06-viz"
test3 = "trial40_M57_F29_070220-viz"
test_directories = [test1, test2, test3]
label_files = [
    "180101_0133S12.csv",
    "180101_0183S34D06.csv",
    "trial40_M57_F29_070220.csv",
]

root = "/Users/wheelerlab/disco_figure_resources/"


def load_and_compute(
    ensemble_directory,
    test_directory,
    label_file,
    iqr_threshold,
    root,
    single_model=False,
):

    data_dict = load_accuracy_metric_data(
        os.path.join(ensemble_directory, test_directory)
    )

    # paint labels onto a 1-D array
    label_array = create_label_array(
        label_csv=os.path.join(root, label_file), spectrogram=data_dict["spectrogram"]
    )

    predictions = data_dict["medians"]

    iqr = np.max(data_dict["iqr"], axis=0)
    predictions[iqr >= iqr_threshold] = cfg.name_to_class_code["BACKGROUND"]
    # for cls in [0, 1]:
    #     print(cls, np.diff(np.where(predictions == cls)))
    # compute total labeled chirps
    label_to_predicted = count_contiguous_regions(predictions)
    label_to_true = count_contiguous_regions(label_array)
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
    # What needs to happen?
    # Well, I need to go through each labeled region.
    (transitions,) = np.where(np.diff(labels) != 0)
    # these are the beginnings and ends.
    chunks = np.concatenate(([0], transitions + 1, [labels.shape[0]]))
    label_to_hit_and_miss = defaultdict(lambda: defaultdict(int))

    for i in range(len(chunks) - 1):
        label_slice = labels[chunks[i] : chunks[i + 1]]
        assert len(np.unique(label_slice)) == 1
        prediction_slice = predictions[chunks[i] : chunks[i + 1]]
        unique = np.unique(prediction_slice)

        if len(unique) == 1:
            if label_slice[0] == prediction_slice[0]:
                # true positive
                label_to_hit_and_miss[cfg.class_code_to_name[label_slice[0]]][
                    "hit"
                ] += 1
            else:
                # false negative
                label_to_hit_and_miss[cfg.class_code_to_name[label_slice[0]]][
                    "miss"
                ] += 1
        else:
            # there are multiple little regions within the predicted slice
            (prediction_transitions,) = np.where(np.diff(prediction_slice) != 0)
            prediction_chunks = np.concatenate(
                ([0], prediction_transitions + 1, [prediction_slice.shape[0]])
            )
            hit = False

            for j in range(len(prediction_chunks) - 1):
                prediction_subslice = prediction_slice[
                    prediction_chunks[j] : prediction_chunks[j + 1]
                ]

                if (
                    prediction_subslice[0] == label_slice[0]
                    and len(prediction_subslice) >= min_length
                ):
                    # this should give us parity, right?
                    hit = True
            if hit:
                label_to_hit_and_miss[cfg.class_code_to_name[label_slice[0]]][
                    "hit"
                ] += 1
            else:
                label_to_hit_and_miss[cfg.class_code_to_name[label_slice[0]]][
                    "miss"
                ] += 1

    return label_to_hit_and_miss


def count_contiguous_regions(labels, min_length=10):
    """
    Count all contiguous classified regions in labels above min_length.
    """
    (transitions,) = np.where(np.diff(labels) != 0)
    # add the 0 boundary on so we capture the first label
    # also add the end so we capture the last label
    boundaries = np.concatenate(([0], transitions + 1, [labels.shape[0]]))
    label_to_count = defaultdict(int)

    for i in range(len(boundaries) - 1):
        # np.diff returns locations adjacent to transitions
        # this goes from 0 -> len()-2
        contig = labels[boundaries[i] : boundaries[i + 1]]

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

    return label_array
