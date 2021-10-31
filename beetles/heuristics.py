import numpy as np
import beetles.inference_utils as infer


def threshold_based_on_mean_uncertainty(predictions, iqr):

    x = 1000
    for i in range(0, predictions.shape[0] - x, x):

        y = np.where(predictions[i : i + x] == infer.NAME_TO_CLASS_CODE["A"])[0]
        z = 0.75 * x
        if len(y) >= z:
            if np.mean(iqr[infer.NAME_TO_CLASS_CODE["A"], i : i + x]) > 0.005:
                predictions[i : i + x] = infer.NAME_TO_CLASS_CODE["BACKGROUND"]
    return predictions


def remove_short_chirps_and_a_chirps_before_b_chirps(predictions, iqr):

    class_idx_to_prediction_start_and_end = infer.aggregate_predictions(predictions)
    starts_and_ends = []
    class_indices = []
    for class_idx, start_and_end in class_idx_to_prediction_start_and_end.items():
        starts_and_ends.extend(start_and_end)
        class_indices.extend([class_idx] * len(start_and_end))

    starts = np.asarray([x[0] for x in starts_and_ends])
    ends = np.asarray([x[1] for x in starts_and_ends])
    sorted_idx = np.argsort(starts)
    starts = starts[sorted_idx]
    ends = ends[sorted_idx]
    class_indices = np.asarray(class_indices)[sorted_idx]
    # now find places where As and Bs have the same end and start, respectively
    for i in range(len(class_indices) - 1):
        if ends[i] - starts[i] <= 15:
            if predictions[ends[i] - 1] == infer.NAME_TO_CLASS_CODE["BACKGROUND"]:
                pass
            elif (
                predictions[ends[i] + 1] == infer.NAME_TO_CLASS_CODE["A"]
                and predictions[starts[i] - 1] == infer.NAME_TO_CLASS_CODE["A"]
            ):
                predictions[starts[i] : ends[i]] = infer.NAME_TO_CLASS_CODE["A"]
            elif (
                predictions[ends[i] + 1] == infer.NAME_TO_CLASS_CODE["B"]
                and predictions[starts[i] - 1] == infer.NAME_TO_CLASS_CODE["B"]
            ):
                predictions[starts[i] : ends[i]] = infer.NAME_TO_CLASS_CODE["B"]
            else:
                predictions[starts[i] : ends[i] + 1] = infer.NAME_TO_CLASS_CODE[
                    "BACKGROUND"
                ]

    for i in range(len(class_indices) - 1):
        if (
            class_indices[i] == infer.NAME_TO_CLASS_CODE["A"]
            and class_indices[i + 1] == infer.NAME_TO_CLASS_CODE["B"]
        ):
            predictions[starts[i] : ends[i]] = infer.NAME_TO_CLASS_CODE["BACKGROUND"]

        # and remove short chirps that aren't in between two As or two Bs

    return predictions


def threshold_bs_on_iqr(predictions, iqr):
    """
    Converts sound classified as B with IQR > iqr_threshold
    to background.
    If classification == b and high A uncertainty, equals B!
    """
    a_idx = infer.NAME_TO_CLASS_CODE["A"]
    iqr_a = iqr[a_idx]
    iqr_threshold = 0.10
    conditional = (iqr_a >= iqr_threshold).astype(bool) & (
        predictions == infer.NAME_TO_CLASS_CODE["B"]
    ).astype(bool)
    predictions[conditional] = infer.NAME_TO_CLASS_CODE["B"]
    return predictions


def threshold_as_on_iqr(predictions, iqr):
    """
    Converts sounds classified as A with IQR > iqr_threshold
    to background.
    if classification == A and high uncertainty b/t and and Background,
    push to background
    """
    iqr_background = iqr[infer.NAME_TO_CLASS_CODE["BACKGROUND"]]
    iqr_a = iqr[infer.NAME_TO_CLASS_CODE["A"]]
    iqr_threshold = 0.10
    conditional = (iqr_background >= iqr_threshold).astype(bool) & (
        predictions == infer.NAME_TO_CLASS_CODE["A"]
    ).astype(bool)
    conditional = (
        conditional
        & (iqr_a >= iqr_threshold).astype(bool)
        & (predictions == infer.NAME_TO_CLASS_CODE["A"]).astype(bool)
    )
    predictions[conditional] = infer.NAME_TO_CLASS_CODE["BACKGROUND"]
    return predictions


# File for holding different heuristics
# The list below contains the heuristics
# that will be applied to the pre-hmm classifications.
# Functions will be applied in the order they are in the list.
HEURISTIC_FNS = [threshold_as_on_iqr, threshold_bs_on_iqr]
