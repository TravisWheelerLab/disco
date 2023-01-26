import logging

import disco_sound.util.inference_utils as infer

log = logging.getLogger(__name__)


def remove_a_chirps_in_between_b_chirps(
    predictions, iqr, name_to_class_code, return_preds=True
):
    """
    :param predictions: np.array (size 1xN) containing point-wise class predictions.
    :param iqr: inter-quartile range of model ensemble.
    :param name_to_class_code: Mapping from name of class to the class code (ex: {"A":2}).
    :param return_preds: bool, default True. Whether or not to return prediction array or a pd.DataFrame containing
    records of begin, end, and sound type.
    :return: prediction array or pd.DataFrame.
    """

    transitions = infer.aggregate_predictions(predictions)
    new_list = []
    for t in transitions:
        x = t.copy()
        if (
            t["end"] - t["start"] <= 20
            and t["class"] != name_to_class_code["BACKGROUND"]
        ):
            predictions[t["start"] : t["end"] + 1] = name_to_class_code["BACKGROUND"]
            x["class"] = name_to_class_code["BACKGROUND"]
            new_list.append(x)

    for i in range(1, len(transitions) - 1):
        current_dct = transitions[i]
        current_class = current_dct["class"]
        if (
            current_class == name_to_class_code["A"]
            and transitions[i - 1]["class"] == name_to_class_code["B"]
            and transitions[i + 1]["class"] == name_to_class_code["B"]
        ):
            log.info(
                f"found an A directly in between two Bs at position {current_dct['start']}. Changing to background."
            )
            current_dct["class"] == name_to_class_code["BACKGROUND"]
            predictions[current_dct["start"] : current_dct["end"]] = name_to_class_code[
                "BACKGROUND"
            ]
        new_list.append(current_dct)

    if return_preds:
        return predictions
    else:
        return new_list


HEURISTIC_FNS = [remove_a_chirps_in_between_b_chirps]
