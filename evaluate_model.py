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

INDEX_TO_LABEL = {0: 'A', 1: 'B', 2: 'X'}


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
    ax[1].set_xticklabels(['{:d}'.format(timepoint+1) for timepoint in timepoints])
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
    generate_confusion_matrix = False
    save_confusion_matrix = False
    predict_not_in_the_wild = False
    predict_in_the_wild = False
    plot_confidences = True
    num_predictions = 30
    mel = True
    log = True
    n_fft = 1024
    vert_trim = 20
    batch_size = num_predictions
    path = None
    random_spect_length = 100

    vert_trim = sa.determine_default_vert_trim(mel, log, n_fft) if None else vert_trim
    spect_type = sa.form_spectrogram_type(mel, n_fft, log, vert_trim)
    if path is None:
        path = 'beetles_cnn_1D_' + spect_type + '.pt'

    # get data based on the booleans set above
    if generate_confusion_matrix or predict_not_in_the_wild:
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
                    spects_list.append([random_file.random_sample_from_entire_spectrogram(random_spect_length, vert_trim),
                                        random_file.name, j])
                print("Appended random samples from", random_file.name + '.')
            i += 1

    # load in trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = tm.CNN1D().to(device)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()

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
                print('Uncorrected accuracy:', round(conf_mat.correct.item()/conf_mat.total, 3))
                print('Corrected accuracy:', round(conf_mat.doctored_correct.item() / conf_mat.doctored_total, 3))
                conf_mat.plot_matrices(classes=INDEX_TO_LABEL.values(), save_images=save_confusion_matrix,
                                       plot_undoctored=True, plot_doctored=True)

        if predict_in_the_wild or plot_confidences:
            # Generates predictions from spectrograms "in the wild"
            for spect, file, idx in spects_list:
                spect = torch.tensor(spect)
                spect = spect.unsqueeze(0)
                spect = spect.to(device)
                output = model(spect)
                pred = output.argmax(dim=1, keepdim=False)
                if not log:
                    spect = spect.log2()
                if predict_in_the_wild:
                    save_sample(image_idx=idx, spectrogram_image=spect, predicted_classes=pred, title=file)
                if plot_confidences:
                    save_confidence_figure(idx=idx, spectrogram_image=spect, predicted_classes=pred,
                                           log_softmaxes=output, title=file)
