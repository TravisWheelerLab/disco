import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from scipy import stats

import disco.cfg as cfg
import disco.util.inference_utils as infer
from disco.util.extract_data import convert_time_to_index, w2s_idx

hop_length = 200
sample_rate = 48000
name_to_class_code = {"A": 0, "B": 1, "X": 2, "BACKGROUND": 2}
ground_truth_data_root = os.path.join("..", "data", "example.csv")
ground_truth_pickle = True


class SoundEvent:
    def __init__(
        self,
        start,
        end,
        ground_truth_array,
        predictions_array,
        avg_iqr_array,
        proportion_event_correct,
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
                self.predictions_mode = 2

        self.span_iqr_average = np.average(avg_iqr_array)
        if self.span_iqr_average > iqr_threshold:
            self.adjusted_preds_label = name_to_class_code["BACKGROUND"]
        else:
            self.adjusted_preds_label = self.predictions_mode


def load_accuracy_metric_data(data_path):
    medians = infer.load_pickle(os.path.join(data_path, "median_predictions.pkl"))
    post_hmm = infer.load_pickle(os.path.join(data_path, "hmm_predictions.pkl"))
    iqr = infer.load_pickle(os.path.join(data_path, "iqrs.pkl"))
    votes = infer.load_pickle(os.path.join(data_path, "votes.pkl"))
    spect = infer.load_pickle(os.path.join(data_path, "raw_spectrogram.pkl"))
    if os.path.isfile(os.path.join(data_path, "raw_preds.pkl")):
        preds = infer.load_pickle(os.path.join(data_path, "raw_preds.pkl"))
        return {
            "raw_preds": preds,
            "spectrogram": spect,
            "medians": medians,
            "post_hmm": post_hmm,
            "iqr": iqr,
            "votes": votes,
        }
    else:
        return {
            "spectrogram": spect,
            "medians": medians,
            "post_hmm": post_hmm,
            "iqr": iqr,
            "votes": votes,
        }


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


def make_sound_events_array(ground_truth, median_argmax, average_iqr):
    sound_event_indices = get_sound_event_indices(ground_truth)

    sound_events = []
    for i in range(len(sound_event_indices)):
        start = sound_event_indices[i]
        if i + 1 <= (len(sound_event_indices) - 1):
            end = sound_event_indices[i + 1]
        else:
            end = len(ground_truth)
        sound_event = SoundEvent(start, end, ground_truth, median_argmax, average_iqr)
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


def eventwise_metrics(data_dict, cov_pcts=None, debug=False):
    """
    Data dict must have a ground_truth key, a spectrogram key, and
    a medians key
    """
    gts = data_dict["ground_truth"]
    preds = data_dict["medians"]
    spectrograms = data_dict["spectrograms"]
    a_recall = []
    b_recall = []
    cmats = []

    if cov_pcts is None:
        cov_pcts = np.linspace(0.1, 0.99, num=10)

    for cov_pct in cov_pcts:

        y_true = []
        y_pred = []
        for j, (ground_truth, pred) in enumerate(zip(gts, preds)):
            if pred.ndim > 1:
                pred = np.argmax(pred, axis=0)

            if len(np.unique(ground_truth)) == 1:
                if np.sum(ground_truth == pred) >= (cov_pct * ground_truth.shape[0]):
                    y_pred.append(ground_truth[0])
                else:
                    y_pred.append(cfg.name_to_class_code["BACKGROUND"])
                    if debug:
                        fig, ax = plt.subplots(nrows=2)
                        ax[0].imshow(spectrograms[j], aspect="auto")
                        ax[1].imshow(
                            pred[
                                np.newaxis,
                            ],
                            aspect="auto",
                            vmin=0,
                            vmax=2,
                        )
                        plt.title(f"truth: {ground_truth[0]}")
                        plt.show()

                y_true.append(ground_truth[0])
            else:
                slice_boundaries = np.where(np.diff(ground_truth) != 0)[0]
                end_idx = (
                    [0] + [x + 1 for x in slice_boundaries] + [ground_truth.shape[0]]
                )
                for i in range(len(end_idx) - 1):

                    gt_slice = ground_truth[end_idx[i] : end_idx[i + 1]]
                    y_true.append(gt_slice[0])
                    assert len(np.unique(gt_slice) == 1)
                    pred_slice = pred[end_idx[i] : end_idx[i + 1]]

                    if np.sum(gt_slice == pred_slice) >= (cov_pct * gt_slice.shape[0]):
                        y_pred.append(gt_slice[0])
                    else:
                        y_pred.append(cfg.name_to_class_code["BACKGROUND"])
                        if debug:
                            fig, ax = plt.subplots(nrows=2)
                            ax[0].imshow(spectrograms[j], aspect="auto")
                            ax[1].imshow(
                                pred[
                                    np.newaxis,
                                ],
                                aspect="auto",
                                vmin=0,
                                vmax=2,
                            )
                            plt.title(f"truth: {ground_truth[0]}")
                            plt.show()

        recalls = metrics.recall_score(y_true, y_pred, average=None)
        cmat = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        a_recall.append(recalls[0])
        b_recall.append(recalls[1])
        cmats.append(cmat)

    return a_recall, b_recall, cmats, cov_pcts


def pointwise_metrics(data_dict):
    # why have we been taking the mean as the uncertainty?
    # i think it makes more sense to grab the max
    iqr = np.max(np.hstack(data_dict["iqr"]), axis=0)
    medians = np.argmax(np.hstack(data_dict["medians"]), axis=0)
    ground_truth = np.hstack(data_dict["ground_truth"])

    precisions = []
    recalls = []
    accuracies = []

    thresholds = np.logspace(-4, 0, 10)[::-1]

    for iqr_threshold in thresholds:
        # if the iqr is greater than this value, then we
        # remove it
        iqr_above = iqr >= iqr_threshold
        # this is what Kayla accomplishes with the `map_to` variable in
        # `adjust_preds_by_confidence`
        medians[iqr_above] = cfg.name_to_class_code["BACKGROUND"]
        accuracies.append(
            metrics.accuracy_score(y_true=ground_truth, y_pred=medians, normalize=True)
        )
        recalls.append(
            metrics.recall_score(y_true=ground_truth, y_pred=medians, average=None)
        )
        precisions.append(
            metrics.precision_score(y_true=ground_truth, y_pred=medians, average=None)
        )

    recalls = np.asarray(recalls)
    precisions = np.asarray(precisions)

    return recalls, precisions, thresholds


def eventwise():
    ten_member = "/Users/mac/beetles_figures/snr_0_ensemble_10_random_init"
    two_member = "/Users/mac/beetles_figures/snr_0_ensemble_2_random_init"
    thirty_member = "/Users/mac/beetles_figures/snr_0_ensemble_30_random_init"

    ten = eventwise_metrics(load_accuracy_metric_pickles(ten_member))
    two = eventwise_metrics(load_accuracy_metric_pickles(two_member))
    thirty = eventwise_metrics(load_accuracy_metric_pickles(thirty_member))

    cov_thresholds = ten[-1]

    fig, ax = plt.subplots(ncols=2, sharey=True, sharex=True)
    # A chirp recall
    ax[0].plot(cov_thresholds, ten[0], label="ten-member ensemble", c="k")
    ax[0].plot(cov_thresholds, two[0], label="two-member ensemble", c="r")
    ax[0].plot(cov_thresholds, thirty[0], label="thirty-member ensemble", c="b")

    ax[0].scatter(cov_thresholds, ten[0], c="k")
    ax[0].scatter(cov_thresholds, two[0], c="r")
    ax[0].scatter(cov_thresholds, thirty[0], c="b")

    ax[1].plot(cov_thresholds, ten[1], c="k")
    ax[1].plot(cov_thresholds, two[1], c="r")
    ax[1].plot(cov_thresholds, thirty[1], c="b")

    ax[1].scatter(cov_thresholds, ten[1], c="k")
    ax[1].scatter(cov_thresholds, two[1], c="r")
    ax[1].scatter(cov_thresholds, thirty[1], c="b")

    ax[0].set_title("a recall")
    ax[1].set_title("b recall")

    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.spines["bottom"].set_color("#808080")
        a.spines["left"].set_color("#808080")

    ax[0].legend()
    alpha = 0.3
    plt.suptitle("event-wise accuracy")

    ax[0].grid(alpha=alpha)
    ax[1].grid(alpha=alpha)
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("accuracy")

    fig.text(0.5, 0.01, "percent event covered", ha="center")

    plt.show()


if __name__ == "__main__":
    # repro figure 1.
    ten_member = f"{os.environ['HOME']}/beetles_figures/snr_0_ensemble_10_random_init"
    two_member = f"{os.environ['HOME']}/beetles_figures/snr_0_ensemble_2_random_init"
    thirty_member = (
        f"{os.environ['HOME']}/beetles_figures/snr_0_ensemble_30_random_init"
    )

    ten = pointwise_metrics(load_accuracy_metric_pickles(ten_member))
    two = pointwise_metrics(load_accuracy_metric_pickles(two_member))
    thirty = pointwise_metrics(load_accuracy_metric_pickles(thirty_member))

    iqr_thresholds = ten[-1]

    fig, ax = plt.subplots(ncols=2, sharey=True, sharex=True)
    # A chirp recall
    (h1,) = ax[0].plot(
        iqr_thresholds, ten[0][:, 0], "o-", label="ten-member ensemble", c="k"
    )
    (h2,) = ax[0].plot(
        iqr_thresholds, two[0][:, 0], "o-", label="two-member ensemble", c="r"
    )
    (h3,) = ax[0].plot(
        iqr_thresholds, thirty[0][:, 0], "o-", label="thirty-member ensemble", c="b"
    )

    # A chirp precision
    ax[0].plot(iqr_thresholds, ten[1][:, 0], "*-", c="k", markersize=10)
    ax[0].plot(iqr_thresholds, two[1][:, 0], "*-", c="r", markersize=10)
    ax[0].plot(iqr_thresholds, thirty[1][:, 0], "*-", c="b", markersize=10)

    # B chirp recall
    ax[1].plot(iqr_thresholds, ten[0][:, 1], "o-", label="ten-member ensemble", c="k")
    ax[1].plot(iqr_thresholds, two[0][:, 1], "o-", label="two-member ensemble", c="r")
    ax[1].plot(
        iqr_thresholds, thirty[0][:, 1], "o-", label="thirty-member ensemble", c="b"
    )

    # B chirp precision
    ax[1].plot(
        iqr_thresholds,
        ten[1][:, 1],
        "*-",
        label="ten-member ensemble",
        c="k",
        markersize=10,
    )
    ax[1].plot(
        iqr_thresholds,
        two[1][:, 1],
        "*-",
        label="two-member ensemble",
        c="r",
        markersize=10,
    )
    ax[1].plot(
        iqr_thresholds,
        thirty[1][:, 1],
        "*-",
        label="thirty-member ensemble",
        c="b",
        markersize=10,
    )

    ax[0].semilogx()
    ax[1].semilogx()

    ax[0].set_title("a chirp precision and recall")
    ax[1].set_title("b chirp precision and recall")

    black_star = mlines.Line2D(
        [],
        [],
        color="black",
        marker="*",
        linestyle="None",
        markersize=10,
        label="precision",
    )

    black_dot = mlines.Line2D(
        [],
        [],
        color="black",
        marker="o",
        linestyle="None",
        markersize=6,
        label="recall",
    )

    ax[0].legend(handles=[h1, h2, h3, black_star, black_dot])
    ax[0].set_ylabel("precision/recall")

    fig.text(0.5, 0.01, "log(iqr_threshold)", ha="center")

    plt.show()
