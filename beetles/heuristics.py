import numpy as np
import beetles.inference_utils as infer

__all__ = ['threshold_as_on_iqr', 'threshold_bs_on_iqr', 'remove_short_chirps_and_a_chirps_before_b_chirps']

def remove_a_chirps_in_between_b_chirps(predictions, iqr, return_preds=True):

    transitions = infer.aggregate_predictions(predictions)
    new_list = []
    for t in transitions:
        x = t.copy()
        if t['end'] - t['start'] <= 20 and t['class'] != infer.NAME_TO_CLASS_CODE['BACKGROUND']:
            predictions[t['start']: t['end']+1] = infer.NAME_TO_CLASS_CODE['BACKGROUND']
            x['class'] = infer.NAME_TO_CLASS_CODE['BACKGROUND']
            new_list.append(x)

    for i in range(1, len(transitions) - 1):
        current_dct = transitions[i]
        current_class = current_dct['class']
        if current_class == infer.NAME_TO_CLASS_CODE['A'] and transitions[i-1]['class'] == infer.NAME_TO_CLASS_CODE['B'] and transitions[i+1]['class'] == infer.NAME_TO_CLASS_CODE['B']:
            print(f"found an A directly in between two Bs f{current_dct['start']}. Changing to background.")
            current_dct['class'] == infer.NAME_TO_CLASS_CODE['BACKGROUND']
            predictions[current_dct['start']:current_dct['end']] = infer.NAME_TO_CLASS_CODE['BACKGROUND']
        new_list.append(current_dct)

    if return_preds:
        return predictions
    else:
        return new_list


HEURISTIC_FNS = [remove_a_chirps_in_between_b_chirps]
