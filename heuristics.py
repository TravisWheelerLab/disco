import inference_utils as infer


def threshold_bs_on_iqr(predictions, iqr, iqr_threshold):
    """
    Converts sound classified as B with IQR > iqr_threshold
    to background.
    """
    b_idx = infer.NAME_TO_CLASS_CODE['B']
    iqr_b = iqr[b_idx]
    predictions[iqr_b >= iqr_threshold] = infer.NAME_TO_CLASS_CODE['BACKGROUND']
    return predictions


def threshold_as_on_iqr(predictions, iqr, iqr_threshold):
    """
    Converts sounds classified as A with IQR > iqr_threshold
    to background.
    """
    a_idx = infer.NAME_TO_CLASS_CODE['A']
    iqr_a = iqr[a_idx]
    predictions[iqr_a >= iqr_threshold] = infer.NAME_TO_CLASS_CODE['BACKGROUND']
    return predictions

