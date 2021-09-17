import torch
import train_model as tm
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import spectrogram_analysis as sa
import random
import pomegranate as pom

INDEX_TO_LABEL = {0: 'A', 1: 'B', 2: 'X'}


def load_in_hmm():
    # TODO: add argument to control the number of sound types
    # and pass it into the HMM
    a_dist = pom.DiscreteDistribution({0: 0.995, 1: 0.00005, 2: 0.00495})
    b_dist = pom.DiscreteDistribution({0: 0.1, 1: 0.88, 2: 0.020})
    x_dist = pom.DiscreteDistribution({0: 0.35, 1: 0.05, 2: 0.60})
    dists = [a_dist, b_dist, x_dist]

    matrix = np.array([[0.995, 0.00000, 0.005],
                       [0.0000, 0.995, 0.005],
                       [0.00001, 0.00049, 0.9995]])

    starts = np.array([0, 0, 1])
    hmm_model = pom.HiddenMarkovModel.from_matrix(matrix, dists, starts)
    hmm_model.bake()

    return hmm_model


def save_sample(image_idx, spectrogram_image, predicted_classes, smoothed_predicted_classes, title=None):
    # saves sample spectrogram with its predicted labels into an image for easier viewing.
    plt.style.use("dark_background")

    if device == 'cuda':
        unsmoothed_spect_image = spectrogram_image.cpu().numpy().squeeze()
        unsmoothed_x = np.arange(len(predicted_classes.squeeze().cpu()))
        unsmoothed_y = np.asarray(predicted_classes[:].cpu()).squeeze() + 2
        unsmoothed_c = predicted_classes.squeeze().cpu()

        smoothed_x = np.arange(len(smoothed_predicted_classes))
        smoothed_y = np.asarray(smoothed_predicted_classes[:]) + 80
        smoothed_c = smoothed_predicted_classes
    else:
        unsmoothed_spect_image = spectrogram_image.numpy().squeeze()
        unsmoothed_x = np.arange(len(predicted_classes.squeeze()))
        unsmoothed_y = np.asarray(predicted_classes[:]).squeeze() + 2
        unsmoothed_c = predicted_classes.squeeze()

        smoothed_x = np.arange(len(smoothed_predicted_classes))
        smoothed_y = np.asarray(smoothed_predicted_classes[:]) + 80
        smoothed_c = smoothed_predicted_classes

    plt.imshow(unsmoothed_spect_image)
    plt.scatter(smoothed_x, smoothed_y, c=smoothed_c, cmap='jet', s=4, vmin=0, vmax=2)
    plt.scatter(unsmoothed_x, unsmoothed_y, c=unsmoothed_c, cmap='jet', s=4, vmin=0, vmax=2)
    plt.colorbar(orientation='horizontal')

    fig_title = "Index and smooth predictions for " + title + " image " + str(image_idx)
    plt.title(fig_title)
    plt.savefig('image_offload/hmm_processed_' + title + '_' + str(image_idx) + '.png')
    plt.close()
    print(title, image_idx, "saved.")


def get_random_slices(data_dir, num_smoothed_preds, mel, n_fft, log, vert_trim, slice_length):
    csvs_and_wav = sa.load_csv_and_wav_files_from_directory(data_dir)
    spects_list = []
    filenames = glob(os.path.join(data_dir, "*"))
    random_file_indices = random.sample(range(0, len(filenames)), 2)
    i = 0
    for filename, (wav, csv) in csvs_and_wav.items():
        if i in random_file_indices:
            spectrogram, label_to_spectrogram = sa.process_wav_file(wav, csv, n_fft, mel, log)
            random_file = sa.BeetleFile(filename, csv, wav, spectrogram, label_to_spectrogram, mel, n_fft, log,
                                        vert_trim)
            for j in range(num_smoothed_preds // 2):
                spects_list.append([random_file.random_sample_from_entire_spectrogram(slice_length, vert_trim),
                                    random_file.name, j])
            print("Appended random samples from", random_file.name + '.')
        i += 1
    return spects_list


if __name__ == '__main__':
    mel = True
    log = True
    n_fft = 900
    vert_trim = 20
    batch_size = 256
    num_smoothed_predictions = 18
    slice_length = 600
    path = "beetles_cnn_1D_mel_log_900_vert_trim_20.pt"
    if vert_trim is None:
        vert_trim = sa.determine_default_vert_trim(mel, log, n_fft)

    # load in pytorch model
    hmm_model = load_in_hmm()

    # grab long snippets
    spects_list = get_random_slices('./wav-files-and-annotations/', num_smoothed_predictions, mel, n_fft, log,
                                    vert_trim, slice_length)
    # load in trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = tm.CNN1D().to(device)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()

    # predict labels and smooth with the HMM
    with torch.no_grad():
        for spect, file, idx in spects_list:
            spect = torch.tensor(spect)
            spect = spect.unsqueeze(0)
            spect = spect.to(device)
            output = model(spect)
            pred = output.argmax(dim=1, keepdim=False)
            if not log:
                spect = spect.log2()
            smoothed_pred = hmm_model.predict(sequence=pred.squeeze().tolist(), algorithm="viterbi")[1:]
            save_sample(idx, spect, pred, smoothed_pred, title=file)
