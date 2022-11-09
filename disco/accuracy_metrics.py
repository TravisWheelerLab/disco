import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from scipy import stats

import disco.inference_utils as infer
from disco.extract_data import convert_time_to_index, w2s_idx

hop_length = 200
sample_rate = 48000
name_to_class_code = {"A": 0, "B": 1, "X": 2, "BACKGROUND": 2}
ground_truth_data_root = os.path.join(".", "data", "example.csv")
ground_truth_pickle = True


class SoundEvent:
    def __init__(
        self,
        start,
        end,
        ground_truth_array,
        predictions_array,
        avg_iqr_array,
        votes_array,
    ):
        self.start = start
        self.end = end

        self.ground_truth_span = ground_truth_array[start:end]
        self.prediction_span = predictions_array[start:end]

        self.ground_truth_label = ground_truth_array[start]
        self.predictions_mode = stats.mode(self.prediction_span, keepdims=True)[0][0]

        self.normalized_accuracy = metrics.accuracy_score(
            y_true=self.ground_truth_span, y_pred=self.prediction_span, normalize=True
        )
        if self.normalized_accuracy >= 0.01 * proportion_event_correct:
            self.correct = True
            self.predictions_mode = self.ground_truth_label
        else:
            self.correct = False
            if self.predictions_mode == self.ground_truth_label:
                self.predictions_mode = 3

        self.span_iqr_average = np.average(avg_iqr_array)
        self.span_votes_average = np.average(votes_array)
        if (self.span_iqr_average > iqr_threshold) or (
            self.span_votes_average < min_votes_needed
        ):
            self.adjusted_preds_label = name_to_class_code["BACKGROUND"]
        else:
            self.adjusted_preds_label = self.predictions_mode


def load_accuracy_metric_pickles(data_path, ground_truth_pickle):
    if ground_truth_pickle:
        ground_truth = infer.load_pickle(os.path.join(data_path, "ground_truth.pkl"))
    else:
        ground_truth = None
    medians = infer.load_pickle(os.path.join(data_path, "median_predictions.pkl"))
    post_hmm = infer.load_pickle(os.path.join(data_path, "hmm_predictions.pkl"))
    iqr = infer.load_pickle(os.path.join(data_path, "iqrs.pkl"))
    votes = infer.load_pickle(os.path.join(data_path, "votes.pkl"))
    average_iqr = np.mean(iqr, axis=0)
    median_argmax = np.argmax(medians, axis=0)
    return ground_truth, median_argmax, post_hmm, average_iqr, votes


def get_ground_truth_np_array(data_path, pickle_shape):
    ground_truth_labels = pd.read_csv(data_path)

    ground_truth_labels["begin samples idx"] = convert_time_to_index(
        ground_truth_labels["Begin Time (s)"], sample_rate
    )
    ground_truth_labels["end samples idx"] = convert_time_to_index(
        ground_truth_labels["End Time (s)"], sample_rate
    )
    ground_truth_labels["begin spect idx"] = [
        w2s_idx(x, hop_length) for x in ground_truth_labels["begin samples idx"]
    ]
    ground_truth_labels["end spect idx"] = [
        w2s_idx(x, hop_length) for x in ground_truth_labels["end samples idx"]
    ]

    # create a continuous numpy array for the ground truth
    ground_truth_label_vector = np.repeat(-1, pickle_shape)

    # fill numpy array with ground truth labels
    for row in range(ground_truth_labels.shape[0]):
        sound_begin = ground_truth_labels["begin spect idx"][row]
        sound_end = ground_truth_labels["end spect idx"][row]
        ground_truth_label_vector[sound_begin:sound_end] = name_to_class_code[
            ground_truth_labels["Sound_Type"].iloc[row]
        ]

    return ground_truth_label_vector


def delete_indices(ground_truth, median_argmax, post_process, average_iqr, votes):
    ground_truth[0] = -1
    indices_to_delete = np.where(ground_truth == -1)

    ground_truth_modified = np.delete(ground_truth, indices_to_delete)
    median_argmax_modified = np.delete(median_argmax, indices_to_delete)
    post_hmm_modified = np.delete(post_process, indices_to_delete)
    average_iqr_modified = np.delete(average_iqr, indices_to_delete)
    votes_modified = np.delete(votes, indices_to_delete, axis=1)
    winning_vote_count = votes_modified[
        median_argmax_modified, np.arange(median_argmax_modified.shape[0])
    ]
    return (
        ground_truth_modified,
        median_argmax_modified,
        post_hmm_modified,
        average_iqr_modified,
        winning_vote_count,
    )


def adjust_preds_by_confidence(
    average_iqr,
    iqr_threshold,
    winning_vote_count,
    min_votes_needed,
    median_argmax,
    map_to=2,
):
    iqr_too_high = np.where(average_iqr > iqr_threshold)
    votes_too_low = np.where(winning_vote_count < min_votes_needed)
    unconfident_indices = np.union1d(iqr_too_high, votes_too_low)
    median_argmax[unconfident_indices] = map_to
    return median_argmax


def make_sound_events_array(
    ground_truth, median_argmax, average_iqr, winning_vote_count
):
    sound_event_indices = get_sound_event_indices(ground_truth)

    sound_events = []
    for i in range(len(sound_event_indices)):
        start = sound_event_indices[i]
        if i + 1 <= (len(sound_event_indices) - 1):
            end = sound_event_indices[i + 1]
        else:
            end = len(ground_truth)
        sound_event = SoundEvent(
            start, end, ground_truth, median_argmax, average_iqr, winning_vote_count
        )
        sound_events.append(sound_event)

    return sound_events


def get_sound_event_indices(ground_truth):
    n_events = (
        1  # note that n_events needs to start at 1 to count the first label as a chunk.
    )
    sound_event_indices = [
        0
    ]  # here, we keep track of where these sound events start and stop
    # (and manually record that the first sound starts at index 0).
    for i in range(len(ground_truth) - 1):
        current_label = ground_truth[i]
        next_label = ground_truth[i + 1]
        if current_label != next_label:
            n_events = n_events + 1
            sound_event_indices.append(i + 1)
    return sound_event_indices


def get_accuracy_metrics(ground_truth, predictions, normalize_confmat=None):
    accuracy = metrics.accuracy_score(
        y_true=ground_truth, y_pred=predictions, normalize=True
    )
    recall = metrics.recall_score(y_true=ground_truth, y_pred=predictions, average=None)
    precision = metrics.precision_score(
        y_true=ground_truth, y_pred=predictions, average=None
    )
    confusion_matrix = metrics.confusion_matrix(
        y_true=ground_truth,
        y_pred=predictions,
        normalize=normalize_confmat,
        labels=[0, 1, 2, 3],
    ).round(3)
    confusion_matrix_nonnorm = metrics.confusion_matrix(
        y_true=ground_truth, y_pred=predictions, labels=[0, 1, 2, 3]
    ).round(4)
    IoU = metrics.jaccard_score(y_true=ground_truth, y_pred=predictions, average=None)
    return accuracy, recall, precision, confusion_matrix, confusion_matrix_nonnorm, IoU


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "infer_data_root", help="where the predicted data is stored.", type=str
    )
    ap.add_argument("out_path", help="where to store the .csv files", type=str)
    ap.add_argument("ensemble_members", type=int)
    ap.add_argument("ensemble_type", type=str)
    args = ap.parse_args()

    ensemble_members = args.ensemble_members
    ensemble_type = args.ensemble_type
    infer_data_root = args.infer_data_root
    out_path = args.out_path

    iqr_threshold_array = [0.05, 0.1, 0.2, 0.4, 1][::-1]

    if ensemble_members == 2:
        min_votes_needed_array = [0, 1]
    elif ensemble_members == 10:
        min_votes_needed_array = [0, 6, 7, 8, 9]
    else:
        min_votes_needed_array = [0, 15, 20, 25, 28, 29]

    pointwise_accuracy_array = [False, True]
    proportion_event_correct_array = [30, 40, 50, 70, 80, 90, 95]

    ensemble_name = str(ensemble_members) + "_" + ensemble_type

    event_wise = {}
    point_wise = {}

    for pointwise_accuracy in pointwise_accuracy_array:
        for iqr_threshold in iqr_threshold_array:
            for min_votes_needed in min_votes_needed_array:
                if iqr_threshold == 1.0 or min_votes_needed == 0:
                    # load in predictions
                    (
                        ground_truth,
                        median_argmax,
                        post_hmm,
                        average_iqr,
                        votes,
                    ) = load_accuracy_metric_pickles(
                        infer_data_root, ground_truth_pickle
                    )
                    # load in ground truth, raven-format labels and create columns with more helpful indexing
                    if not ground_truth_pickle:
                        ground_truth = get_ground_truth_np_array(
                            ground_truth_data_root, median_argmax.shape[0]
                        )
                    # remove indices where ground truth csv has no labels
                    (
                        ground_truth,
                        median_argmax,
                        post_hmm,
                        average_iqr,
                        winning_vote_count,
                    ) = delete_indices(
                        ground_truth,
                        median_argmax,
                        post_hmm,
                        average_iqr,
                        votes,
                    )

                    if pointwise_accuracy:
                        final_adjusted_preds = adjust_preds_by_confidence(
                            average_iqr,
                            iqr_threshold,
                            winning_vote_count,
                            min_votes_needed,
                            median_argmax,
                        )
                        (
                            accuracy,
                            recall,
                            precision,
                            confusion_matrix,
                            confmat_nonnorm,
                            IoU,
                        ) = get_accuracy_metrics(
                            ground_truth,
                            final_adjusted_preds,
                            normalize_confmat="true",
                        )

                        large_dict = defaultdict(dict)
                        category_string = f"{iqr_threshold}/{min_votes_needed}"
                        print("pointwise", category_string)

                        large_dict["accuracy"] = accuracy
                        large_dict["recall, A chirp"] = recall[0]
                        large_dict["recall, B chirp"] = recall[1]
                        large_dict["precision, A chirp"] = precision[0]
                        large_dict["precision, B chirp"] = precision[1]
                        large_dict["IoU, A chirp"] = IoU[0]
                        large_dict["IoU, B chirp"] = IoU[1]

                        large_dict["confusion_matrix"] = confusion_matrix
                        large_dict["confusion_matrix_nonnorm"] = confmat_nonnorm
                        point_wise[category_string] = large_dict

                    else:
                        for proportion_event_correct in proportion_event_correct_array:
                            sound_events = make_sound_events_array(
                                ground_truth,
                                median_argmax,
                                average_iqr,
                                winning_vote_count,
                            )

                            event_ground_truth = [
                                sound_event.ground_truth_label
                                for sound_event in sound_events
                            ]
                            event_preds_uncertainty_adjusted = [
                                sound_event.adjusted_preds_label
                                for sound_event in sound_events
                            ]
                            event_preds_unadjusted = [
                                sound_event.predictions_mode
                                for sound_event in sound_events
                            ]

                            (
                                accuracy,
                                recall,
                                precision,
                                confusion_matrix,
                                confmat_nonnorm,
                                IoU,
                            ) = get_accuracy_metrics(
                                event_ground_truth,
                                event_preds_uncertainty_adjusted,
                                normalize_confmat="true",
                            )

                            large_dict = defaultdict(dict)
                            large_dict["accuracy"] = accuracy
                            large_dict["recall, A chirp"] = recall[0]
                            large_dict["recall, B chirp"] = recall[1]

                            large_dict["A chirp correct"] = confmat_nonnorm[0, 0]
                            large_dict["A chirp incorrect"] = np.sum(
                                confmat_nonnorm[0, 1:]
                            )
                            large_dict["B chirp correct"] = confmat_nonnorm[1, 1]

                            large_dict["B chirp incorrect"] = (
                                confmat_nonnorm[1, 0]
                                + confmat_nonnorm[1, 2]
                                + confmat_nonnorm[1, 3]
                            )

                            large_dict["confusion_matrix"] = confusion_matrix
                            large_dict["confusion_matrix_nonnorm"] = confmat_nonnorm
                            category_string = f"{iqr_threshold}/{min_votes_needed}/{proportion_event_correct}"
                            print("event_wise", category_string)
                            event_wise[
                                f"{iqr_threshold}/{min_votes_needed}/{proportion_event_correct}"
                            ] = large_dict

    print(
        "Saving dataframe to "
        + os.path.join(
            out_path,
            f"~~point OR event~~_wise_{ensemble_members}_{ensemble_type}.csv",
        )
    )
    os.makedirs(out_path, exist_ok=True)

    pointwise_dataframe = pd.DataFrame.from_dict(point_wise)
    eventwise_dataframe = pd.DataFrame.from_dict(event_wise)

    pointwise_dataframe.to_csv(
        os.path.join(out_path, f"point_wise_{ensemble_members}_{ensemble_type}.csv")
    )
    eventwise_dataframe.to_csv(
        os.path.join(
            out_path,
            f"event_wise_{ensemble_members}_{ensemble_type}.csv",
        )
    )
