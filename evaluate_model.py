from data_feeder import SpectrogramDataset
import torch
import train_model as tm
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import confusion_matrix as cm
import spectrogram_analysis as sa
import random
from scipy import stats

INDEX_TO_LABEL = {0: 'A', 1: 'B', 2: 'X'}


class EnsembleEvaluator:

    def __init__(self, ensemble, spect, file):
        self.ensemble = ensemble
        self.spect = spect
        self.length_of_spect = spect.shape[-1]
        self.all_softmaxes, self.all_predictions, self.number_of_classes, = get_softmaxes_and_preds(ensemble, spect)
        self.iqrs, self.iqrs_added = get_iqrs(self.all_softmaxes, self.number_of_classes, self.length_of_spect)
        self.modes = stats.mode(self.all_predictions)[0].squeeze().astype(int)
        self.medians_of_softmaxes = np.median(self.all_softmaxes, axis=1)
        self.ensemble_pred = np.argmax(self.medians_of_softmaxes, axis=0)
        self.means_of_softmaxes = np.mean(self.all_softmaxes, axis=1)
        self.median_confidences_of_winner, self.mean_confidences_of_winner = get_mean_and_median_confs(
            self.medians_of_softmaxes, self.means_of_softmaxes, self.modes, self.length_of_spect)
        self.filename = file

    def save_image(self, image_number, overlay_preds=True, overlay_iqrs=True,
                                overlay_all_median_softmaxes=True):
        plt.style.use("dark_background")

        if device == 'cuda':
            self.spect = self.spect.cpu()
        self.spect = self.spect.numpy().squeeze()
        plt.imshow(self.spect)

        if overlay_preds:
            cmap = 'jet'
            x = np.arange(len(self.ensemble_pred))
            y = np.zeros(self.ensemble_pred.shape[0]) + 2
            c = self.ensemble_pred
            plt.scatter(x, y, c=c, cmap=cmap, s=4, vmin=0, vmax=2)
            fig_title = "predictions for image " + str(image_number)
            plt.title(fig_title)

        if overlay_iqrs:
            cmap = 'plasma'
            x = np.arange(len(self.iqrs_added))
            y = np.full(shape=len(self.iqrs_added), fill_value=self.spect.shape[0] - 60)
            c = self.iqrs_added
            plt.scatter(x, y, c=c, cmap=cmap, s=4, vmin=0, vmax=1.3)
            # plt.colorbar(location='bottom')

        if overlay_all_median_softmaxes:
            cmap = 'plasma'
            for smax_class in range(self.medians_of_softmaxes.shape[0]):
                x = np.arange(self.medians_of_softmaxes.shape[-1])
                y = np.full(shape=self.medians_of_softmaxes.shape[-1],
                            fill_value=self.spect.shape[0] - (15 - (smax_class + 1) * 5))
                c = self.medians_of_softmaxes[smax_class]
                plt.scatter(x, y, c=c, cmap=cmap, s=4, vmin=0, vmax=1)
            plt.colorbar(location='bottom')

        plt.savefig('image_offload/ensemble_preds_' + file + '_' + str(image_number) + '.png')
        plt.close()
        print("Image {} saved.".format(image_number))


def get_mean_and_median_confs(medians_of_softmaxes, means_of_softmaxes, modes, length_of_spect):
    median_confidences_of_winning_labels = np.zeros(shape=length_of_spect)
    mean_confidences_of_winning_labels = np.zeros(shape=length_of_spect)
    # make a median softmax for each class value
    # need an array that is 3x20 for this

    for i in range(length_of_spect):
        median_confidences_of_winning_labels[i] = medians_of_softmaxes[modes[i]][i]
        mean_confidences_of_winning_labels[i] = means_of_softmaxes[modes[i]][i]
    # use modes value to index into it to get the final array
    return mean_confidences_of_winning_labels, median_confidences_of_winning_labels


def get_iqrs(all_softmaxes, number_of_classes, length_of_spect):
    # finds the interquartile range for the softmax values given by all of the models for each class,
    # index-wise over the entire spectrogram.
    iqrs = np.zeros((number_of_classes, length_of_spect))
    for iqr_class in range(iqrs.shape[0]):
        for spect_idx in range(length_of_spect):
            q75, q25 = np.percentile(all_softmaxes[iqr_class][:, spect_idx], [75, 25])
            iqrs[iqr_class][spect_idx] = q75 - q25
    # iqrs_added takes a sum over the iqrs of each class. So the IQR for the 0th index of the
    # spectrogram is the iqr of the 0th class + the 1st class + the 2nd class, and so on,
    # which tells us the overall certainty for all 10 models altogether for that individual timepoint.
    iqrs_added = np.sum(iqrs, axis=0)
    return iqrs, iqrs_added


def get_softmaxes_and_preds(ensemble, spect):
    for index, model in enumerate(ensemble):
        model.eval()
        output = model(spect)
        number_of_classes = output.shape[1]
        length_of_spect = output.shape[-1]
        pred = output.argmax(dim=1, keepdim=False).squeeze().cpu().numpy()
        softmaxes = np.exp(output.squeeze().cpu()).numpy()
        if index == 0:
            # indexing for reference:
            # all_softmaxes[class][model (index)][spectrogram index]
            # all_predictions[model (index)][spectrogram index]
            all_softmaxes = np.zeros(shape=(number_of_classes, len(models), length_of_spect))
            all_predictions = np.zeros(shape=(len(models), length_of_spect))

        # add these softmaxes and predictions to the multidimensional numpy arrays
        for label in range(softmaxes.shape[0]):
            all_softmaxes[label][index] = softmaxes[label]
        all_predictions[index] = pred

    all_preds_int = all_predictions.astype(int)
    return all_softmaxes, all_preds_int, number_of_classes


def save_confidence_figure(idx, spectrogram_image, predicted_classes, log_softmaxes, title):
    plt.style.use("dark_background")
    softmaxes = np.exp(log_softmaxes).squeeze()
    fig, ax = plt.subplots(nrows=2, ncols=1)

    if device == 'cuda':
        spectrogram_image = spectrogram_image.cpu().numpy().squeeze()
        x = np.arange(len(predicted_classes.squeeze().cpu()))
        y = np.asarray(predicted_classes[:].cpu()).squeeze() + 2
        c = predicted_classes.squeeze().cpu()
    else:
        x = np.arange(len(predicted_classes.squeeze()))
        y = np.asarray(predicted_classes[:]).squeeze() + 2
        c = predicted_classes.squeeze()

    ax[0].imshow(spectrogram_image.squeeze())
    ax[0].scatter(x, y, c=c, cmap='jet', s=4, vmin=0, vmax=2)

    im = ax[1].imshow(softmaxes, interpolation='nearest', cmap='plasma', vmin=0, vmax=1)
    ax[1].set_yticks(range(softmaxes.shape[0]))
    ax[1].set_yticklabels(['0', '1', '2'])
    timepoints = np.array(range(0, softmaxes.shape[-1], 50))
    ax[1].set_xticks(timepoints)
    ax[1].set_xticklabels(['{:d}'.format(timepoint + 1) for timepoint in timepoints])
    ax[1].set_xlabel('Spectrogram Index')
    ax[1].set_title('Softmaxes')
    cbar = fig.colorbar(ax=ax[1], mappable=im, orientation='horizontal')
    cbar.set_label('Confidence')
    filename = 'softmaxes_' + title + '_' + str(idx) + '.png'
    plt.savefig('image_offload/' + filename)
    print(filename, 'saved.')
    plt.close(fig)


def save_sample(image_idx, spectrogram_image, predicted_classes, title=None):
    # saves sample spectrogram with its predicted labels into an image for easier viewing.
    plt.style.use("dark_background")
    cmap = 'jet'

    if device == 'cuda':
        spect_image = spectrogram_image.cpu().numpy().squeeze()
        x = np.arange(len(predicted_classes.squeeze().cpu()))
        y = np.asarray(predicted_classes[:].cpu()).squeeze() + 2
        c = predicted_classes.squeeze().cpu()
    else:
        spect_image = spectrogram_image.numpy().squeeze()
        x = np.arange(len(predicted_classes.squeeze()))
        y = np.asarray(predicted_classes[:]).squeeze() + 2
        c = predicted_classes.squeeze()

    plt.imshow(spect_image)
    plt.scatter(x, y, c=c, cmap=cmap, s=4, vmin=0, vmax=2)
    if title is None:
        title = "pre-labeled-sample"
    fig_title = "predictions for " + title + " image " + str(image_idx)
    plt.title(fig_title)
    plt.savefig('image_offload/' + title + '_' + str(image_idx) + '.png')
    plt.close()
    print(title, image_idx, "saved.")


if __name__ == '__main__':
    model_ensemble = True
    generate_confusion_matrix = False
    save_confusion_matrix = False
    predict_not_in_the_wild = False
    predict_in_the_wild = True
    plot_confidences = False
    plot_IQR_confidences = False
    num_predictions = 30
    mel = True
    log = True
    n_fft = 900
    vert_trim = None
    batch_size = num_predictions
    path = None
    random_spect_length = 500
    if vert_trim is None:
        vert_trim = sa.determine_default_vert_trim(mel, log, n_fft)
    spect_type = sa.form_spectrogram_type(mel, n_fft, log, vert_trim)
    if path is None:
        if model_ensemble:
            path = os.path.join('models', spect_type + '_ensemble')
            models = glob(os.path.join(path, '*'))
        else:
            models = [os.path.join('models', 'beetles_cnn_1D_' + spect_type + '.pt')]
    else:
        models = path

    # load in trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load in trained models
    ensemble = []
    for model_path in models:
        model = tm.CNN1D().to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        ensemble.append(model)

    if generate_confusion_matrix or predict_not_in_the_wild:
        # get data based on the booleans set above
        # creates  dataloader object and conf matrix to be used for statistical evaluation
        test_dataset = SpectrogramDataset(dataset_type='test',
                                          spect_type=spect_type,
                                          clip_spects=False,
                                          batch_size=num_predictions)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
        conf_mat = cm.ConfusionMatrix(num_classes=3)

    if predict_in_the_wild or plot_confidences:
        data_dir = './wav-files-and-annotations/'
        csvs_and_wav = sa.load_csv_and_wav_files_from_directory(data_dir)
        spects_list = []

        filenames = glob(os.path.join(data_dir, "*"))
        random_file_indices = random.sample(range(0, len(filenames)), 2)

        i = 0
        for filename, (wav, csv) in csvs_and_wav.items():
            # generate 12 random samples from wav files
            if i in random_file_indices:
                spectrogram, label_to_spectrogram = sa.process_wav_file(wav, csv, n_fft, mel, log)
                random_file = sa.BeetleFile(filename, csv, wav, spectrogram, label_to_spectrogram, mel, n_fft, log,
                                            vert_trim=vert_trim)
                for j in range(num_predictions // 2):
                    spects_list.append(
                        [random_file.random_sample_from_entire_spectrogram(random_spect_length, vert_trim),
                         random_file.name, j])
                print("Appended random samples from", random_file.name + '.')
            i += 1

    with torch.no_grad():
        if generate_confusion_matrix or predict_not_in_the_wild:
            i = 0
            for data, target in test_loader:
                if i < num_predictions:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
                    if predict_not_in_the_wild:
                        if not log:
                            data = data.log2()
                        save_sample(image_idx=i, spectrogram_image=data, predicted_classes=pred)
                    is_a = True if target[0][0].item() == 0 else False

                    if generate_confusion_matrix:
                        conf_mat.increment(target, pred, device, is_a)
                i += 1

            if generate_confusion_matrix:
                print('Uncorrected accuracy:', round(conf_mat.correct.item() / conf_mat.total, 3))
                print('Corrected accuracy:', round(conf_mat.doctored_correct.item() / conf_mat.doctored_total, 3))
                conf_mat.plot_matrices(classes=INDEX_TO_LABEL.values(), save_images=save_confusion_matrix,
                                       plot_undoctored=True, plot_doctored=True)

        if predict_in_the_wild or plot_confidences:
            # Generates predictions from spectrograms "in the wild"
            for spect, file, idx in spects_list:
                spect = torch.tensor(spect)
                spect = spect.unsqueeze(0)
                spect = spect.to(device)
                if model_ensemble:
                    ensemble_eval = EnsembleEvaluator(ensemble, spect, file)
                else:
                    output = model(spect)
                    pred = output.argmax(dim=1, keepdim=False)
                    if not log:
                        spect = spect.log2()
                    if predict_in_the_wild:
                        save_sample(image_idx=idx, spectrogram_image=spect, predicted_classes=pred, title=file)
                    if plot_confidences:
                        save_confidence_figure(idx=idx, spectrogram_image=spect, predicted_classes=pred,
                                               log_softmaxes=output, title=file)
