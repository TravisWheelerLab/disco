from glob import glob
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from beetles.dataset import SpectrogramDatasetMultiLabel
import beetles.inference_utils as iu
import beetles.heuristics as hu

def get_max_classification_in_region(arr):
    un, cnt = np.unique(arr, return_counts=True)
    return un[np.argmax(cnt)]

# 6 models out of 10 vote with a probability of 0.5
# a pessimistic median

if __name__ == '__main__':

    test = '/home/tc229954/data/beetles/extracted_data/test/mel_no_log_1150_no_vert_trim/*'
    test_files = glob(test) # + glob(test.replace('test', 'validation')) + glob(test.replace('test', 'train'))

    dset = SpectrogramDatasetMultiLabel(test_files,
                                        vertical_trim=20)

    ensemble = iu.assemble_ensemble(model_directory='/home/tc229954/.cache/beetles',
                                    model_extension='*pt',
                                    device='cuda',
                                    in_channels=108)

    dset = torch.utils.data.DataLoader(dset,
                                    batch_size=1)

    median_argmax_cmat = np.zeros((3, 3))
    median_post_threshold_cmat = np.zeros((3, 3))
    median_post_hmm_cmat = np.zeros((3, 3))

    all_predictions = []
    all_features = []
    all_iqrs = []
    all_labels = []

    with torch.no_grad():

        for features, labels in dset:
            features = features.to('cuda')
            labels = labels.numpy().squeeze()
            if len(np.unique(labels)) > 1:

                preds = np.asarray(iu.predict_with_ensemble(ensemble, features))
                medians, iqrs = iu.calculate_median_and_iqr(preds)
                medians = medians.squeeze()
                iqrs = iqrs.squeeze()

                all_features.append(features.cpu().numpy().squeeze())

                all_predictions.append(medians)
                all_iqrs.append(iqrs)
                all_labels.append(labels)

    medians = np.column_stack(all_predictions)
    features = np.column_stack(all_features)
    iqrs = np.column_stack(all_iqrs)
    labels = np.concatenate(all_labels)
    trans = np.where(np.diff(labels) != 0)[0]

    median_argmax = np.argmax(medians, axis=0)

    post_threshold = median_argmax.copy()
    post_threshold = hu.remove_a_chirps_in_between_b_chirps(post_threshold, post_threshold)
    post_hmm = iu.smooth_predictions_with_hmm(median_argmax.copy())
    cmat_arg = confusion_matrix(labels, median_argmax)
    cmat_hmm = confusion_matrix(labels, post_hmm)
    cmat_threshold = confusion_matrix(labels, post_threshold)
    print(cmat_arg)

    print(cmat_arg/ np.sum(cmat_arg, axis=0, keepdims=True))
    print(cmat_arg/ np.sum(cmat_arg, axis=1, keepdims=True))

    start = 0

    median_classifications = []
    post_threshold_classifications = []
    post_hmm_classifications = []
    labels_classifications = []


    coverage_threshold = 0.5

    for i in range(len(trans)):
        end = int(trans[i])
        region = labels[start+1:end]
        true_label = get_max_classification_in_region(region)
        labels_classifications.append(true_label)

        median_argmax_region = median_argmax[start+1: end]
        post_threshold_region = post_threshold[start+1: end]
        post_hmm_region = post_hmm[start+1: end]

        if np.sum(median_argmax_region == true_label) / len(median_argmax_region) >= coverage_threshold:
            median_classifications.append(true_label)
        else:
            median_argmax_region = median_argmax_region[median_argmax_region != true_label]
            median_classifications.append(get_max_classification_in_region(median_argmax_region))

        if np.sum(post_threshold_region == true_label) / len(post_threshold_region) >= coverage_threshold:
            post_threshold_classifications.append(true_label)
        else:
            post_threshold_region = post_threshold_region[post_threshold_region != true_label]
            post_threshold_classifications.append(get_max_classification_in_region(post_threshold_region))

        if np.sum(post_hmm_region == true_label) / len(post_hmm_region) >= coverage_threshold:
            post_hmm_classifications.append(true_label)
        else:
            post_hmm_region = post_hmm_region[post_hmm_region != true_label]
            post_hmm_classifications.append(get_max_classification_in_region(post_hmm_region))

        start = end

m = confusion_matrix(labels_classifications, median_classifications)
pt = confusion_matrix(labels_classifications, post_threshold_classifications)
ph = confusion_matrix(labels_classifications, post_hmm_classifications)

print(m)
print(m/ np.sum(m,axis=0, keepdims=True) )
print(m/ np.sum(m, axis=1, keepdims=True) )
