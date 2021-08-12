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
    plt.scatter(x, y, c=c, cmap=cmap, s=4)
    plt.scatter(x, y+15, c=c, cmap=cmap, s=4)
    plt.colorbar()
    if title is None:
        title = "pre-labeled-sample"
    fig_title = "Smoothed predictions for " + title + " image " + str(image_idx)
    plt.title(fig_title)
    plt.savefig('image_offload/' + title + '_' + str(image_idx) + '.png')
    plt.close()
    print(title, image_idx, "saved.")


if __name__ == '__main__':

    generate_confusion_matrix = False
    save_confusion_matrix = False
    predict_not_in_the_wild = False
    predict_in_the_wild = True
    plot_confidences = False
    num_predictions = 5
    mel = True
    log = False
    n_fft = 800
    vert_trim = 30
    batch_size = num_predictions
    path = "beetles_cnn_1D_mel_no_log_800_vert_trim_30_600_epochs.pt"

    vert_trim = sa.determine_default_vert_trim(mel, log, n_fft) if None else vert_trim
    spect_type = sa.form_spectrogram_type(mel, n_fft, log, vert_trim)

    # get data based on the booleans set above
    if generate_confusion_matrix:
        # creates  dataloader object and conf matrix to be used for statistical evaluation
        root = os.path.join('data/test', 'spect')
        files = glob(os.path.join(root, "*"))
        test_dataset = SpectrogramDataset(dataset_type='test',
                                          spect_type=spect_type,
                                          clip_spects=False,
                                          batch_size=num_predictions)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
        conf_mat = cm.ConfusionMatrix(num_classes=3)

    if predict_in_the_wild:
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
                    spects_list.append([random_file.random_sample_from_entire_spectrogram(300, vert_trim),
                                        random_file.name, j])
                print("Appended random samples from", random_file.name + '.')
            i += 1

    # load in trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = tm.CNN1D().to(device)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()

    with torch.no_grad():
        if generate_confusion_matrix:
            # Assesses accuracy of the model
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
                if predict_not_in_the_wild:
                    if not log:
                        data = data.log2()
                    if plot_confidences:
                        save_confidence_figure(spectrogram_image=data, predicted_classes=pred, softmaxes=output)
                    else:
                        save_sample(image_idx=0, spectrogram_image=data, predicted_classes=pred)

                is_a = True if target[0][0].item() == 0 else False

                conf_mat.increment(target, pred, device, is_a)

            print('Uncorrected accuracy:', round(conf_mat.correct.item()/conf_mat.total, 3))
            print('Corrected accuracy:', round(conf_mat.doctored_correct.item() / conf_mat.doctored_total, 3))
            conf_mat.plot_matrices(classes=INDEX_TO_LABEL.values(), save_images=save_confusion_matrix,
                                   plot_undoctored=True, plot_doctored=True)

        if predict_in_the_wild:
            # Generates predictions from spectrograms "in the wild"
            for spect, file, idx in spects_list:
                spect = torch.tensor(spect)
                spect = spect.unsqueeze(0)
                spect = spect.to(device)
                output = model(spect)
                pred = output.argmax(dim=1, keepdim=False)
                if not log:
                    spect = spect.log2()
                save_sample(image_idx=idx, spectrogram_image=spect, predicted_classes=pred, title=file)
