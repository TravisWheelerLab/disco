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

    if device == 'cuda':
        plt.imshow(spectrogram_image.cpu().numpy().squeeze())
        plt.scatter(np.arange(len(predicted_classes.squeeze().cpu())),
                    np.asarray(predicted_classes[:].cpu()).squeeze() + 2,
                    c=predicted_classes.squeeze().cpu(),
                    cmap='jet',
                    s=4)
    else:
        plt.imshow(spectrogram_image.numpy().squeeze())
        plt.scatter(np.arange(len(predicted_classes.squeeze())),
                    np.asarray(predicted_classes[:]).squeeze() + 2,
                    c=predicted_classes.squeeze(),
                    cmap='jet',
                    s=4)
    plt.colorbar()
    if title is None:
        title = "pre-labeled-sample"
    fig_title = "Index predictions for " + title + " image " + str(image_idx)
    plt.title(fig_title)
    plt.savefig('image_offload/' + title + str(image_idx) + '.png')
    plt.close()
    print(title, image_idx, "saved.")


if __name__ == '__main__':

    # choices to make when assessing the model:
    # 1. create the confusion matrices
    # 2. predict pre-labeled areas
    # 3. predict random unlabeled areas
    generate_confusion_matrix = False
    # the next boolean will generally stay false unless it is desired to assess accuracy of pre-labeled sounds
    # generate_confusion_matrix needs to be true for not_in_the_wild predictions to run.
    predict_not_in_the_wild = False
    predict_in_the_wild = True
    num_predictions = 20
    mel = True
    log = True
    n_fft = 1600

    spect_type = sa.form_spectrogram_type(mel, n_fft, log)

    # get data based on the booleans set above
    if generate_confusion_matrix:
        # creates  dataloader object and conf matrix to be used for statistical evaluation
        root = os.path.join('data/test', 'spect')
        files = glob(os.path.join(root, "*"))
        test_loader = torch.utils.data.DataLoader(SpectrogramDataset(dataset_type='test', spect_type=spect_type,
                                                                     clip_spects=False),
                                                  batch_size=1,
                                                  shuffle=True)
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
                random_file = sa.BeetleFile(filename, csv, wav, spectrogram, label_to_spectrogram, mel, n_fft, log)
                for j in range(num_predictions // 2):
                    spects_list.append([random_file.random_sample_from_entire_spectrogram(300), random_file.name, j])
                print("Appended random samples from", random_file.name + '.')
            i += 1

    # load in trained model
    path = "beetles_cnn_1D_mel_log_1600_no_vert_trim.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = tm.CNN1D().to(device)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()

    with torch.no_grad():
        if generate_confusion_matrix:
            # Assesses accuracy of the model
            i = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
                if i < num_predictions and predict_not_in_the_wild:
                    save_sample(image_idx=i, spectrogram_image=data, predicted_classes=pred)
                i += 1
                conf_mat.increment(target, pred, device)
            conf_mat.plot(classes=INDEX_TO_LABEL.values(), save_images=True)
        if predict_in_the_wild:
            # Generates predictions from spectrograms "in the wild"
            for spect, file, idx in spects_list:
                spect = torch.tensor(spect)  #.unsqueeze(0).unsqueeze(0)
                breakpoint()
                spect = spect.to(device)
                breakpoint()
                output = model(spect)
                pred = output.argmax(dim=1, keepdim=False)
                save_sample(image_idx=idx, spectrogram_image=spect, predicted_classes=pred, title=file)
