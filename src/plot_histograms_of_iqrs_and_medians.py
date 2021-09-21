import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
from sklearn.metrics import confusion_matrix

import inference_utils
from data_feeder import SpectrogramDataset
from spectrogram_analysis import form_spectrogram_type

if __name__ == '__main__':
    mel = True
    n_fft = 800
    log = True
    vert_trim = 30

    spect_type = form_spectrogram_type(mel, n_fft, log, vert_trim)
    train_dataset = SpectrogramDataset(dataset_type='test',
                                       spect_type=spect_type,
                                       clip_spects=False,
                                       batch_size=1,
                                       bootstrap_sample=False)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1)
    ensemble = inference_utils.assemble_ensemble('./models/mel_log_800_vert_trim_30_ensemble/',
                                                 '.pt', 'cuda', 98)

    class_to_prediction_medians = defaultdict(list)
    class_to_prediction_iqrs = defaultdict(list)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cmat = np.zeros((3, 3))
    with torch.no_grad():
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.cpu().numpy()
            label = np.unique(labels)[0]

            ensemble_preds = np.stack(inference_utils.predict_with_ensemble(ensemble, features))
            iqrs, medians = inference_utils.calculate_median_and_iqr(ensemble_preds)
            hmm_smoothed = inference_utils.run_hmm(np.argmax(medians.squeeze(), axis=0))
            cmat += confusion_matrix(labels.squeeze(), hmm_smoothed, labels=[0, 1, 2])
            medians = medians.squeeze()
            iqrs = iqrs.squeeze()
            class_to_prediction_iqrs[label].append(iqrs.squeeze())
            class_to_prediction_medians[label].append(medians.squeeze())

    precision = np.diag(cmat/np.sum(cmat, axis=1))
    recall = np.diag(cmat/np.sum(cmat, axis=0))
    print('precision and recall of ensemble classifier:')
    print(precision)
    print(recall)
    exit()

    fig, ax = plt.subplots(ncols=2, nrows=3, sharex=True, sharey=True, figsize=(13, 10))
    for i, class_idx in enumerate(inference_utils.CLASS_CODE_TO_NAME.keys()):
        # each class has a hist of values and iqrs
        medians = class_to_prediction_medians[class_idx]
        iqrs = class_to_prediction_iqrs[class_idx]
        preds = np.concatenate(medians, axis=-1)
        uncertainties = np.concatenate(iqrs, axis=-1)
        for cls in inference_utils.CLASS_CODE_TO_NAME.keys():
            ax[i, 0].hist(preds[cls], histtype='step', label=inference_utils.CLASS_CODE_TO_NAME[cls])
            ax[i, 1].hist(uncertainties[cls], histtype='step', label=inference_utils.CLASS_CODE_TO_NAME[cls])

        ax[i, 1].legend()
        ax[i, 0].legend()
        ax[i, 1].set_title('median softmax prob. for data known to be {}'.format(inference_utils.CLASS_CODE_TO_NAME[class_idx]))
        ax[i, 0].set_title('iqr of softmax prob. for data known to be {}'.format(inference_utils.CLASS_CODE_TO_NAME[class_idx]))

    s = 'uncertainties and median predictions for each labeled example'
    ax[-1, 1].set_xlabel('softmax output')
    ax[-1, 0].set_xlabel('softmax output')
    plt.suptitle(s)
    plt.savefig('/home/tc229954/histograms_of_medians_and_iqrs.png')
    plt.close()
