import time
import random
import torch
import torchaudio
import numpy as np
import pandas as pd
import pickle
import os

from glob import glob
from sklearn.model_selection import train_test_split


def w2s_idx(idx, hop_length):
    """
    Converts .wav file index to spectrogram index.
    :param idx: Index to convert.
    :param hop_length: Length between subsequent spectrogram windows.
    :return:
    """
    return idx // hop_length


def create_label_to_spectrogram(
    spect,
    labels,
    hop_length,
    name_to_class_code,
    excluded_classes,
    neighbor_tolerance=100,
):
    """
    Accepts a spectrogram (torch.Tensor) and labels (pd.DataFrame) and returns
    a song type and a list of tensors of those spectrograms as a value. e.g.:

    len(label_to_spectrogram['A']) = 49 (number of sounds of this subtype)
    type(label_to_spectrogram['A']) = class 'list'
    type(label_to_spectrogram['A'][0]) = class 'torch.Tensor'

    If two labeled regions are within :param: neighbor_tolerance of one another they'll be considered
    one example and saved as one array.

    :return: list: List of lists. Each sublist contains a feature tensor and the associated vector of labels.
    """

    labels["begin spect idx"] = [w2s_idx(x, hop_length) for x in labels["begin idx"]]
    labels["end spect idx"] = [w2s_idx(x, hop_length) for x in labels["end idx"]]

    contiguous_indices = []
    if labels.shape[0] == 1:
        contiguous_indices.append([0])
    else:
        labels = labels.sort_values(by="begin idx")
        i = 0
        while i < labels.shape[0] - 1:
            contig = [i]
            while (
                labels.iloc[i + 1]["begin idx"] - labels.iloc[i]["end idx"]
            ) <= neighbor_tolerance:
                contig.extend([i + 1])
                i += 1
                if i == labels.shape[0] - 1:
                    break
            if i == labels.shape[0] - 2 and i + 1 not in contig:
                contiguous_indices.append([i + 1])
            contiguous_indices.append(contig)
            i += 1

    features_and_labels = []
    for contig in contiguous_indices:
        contiguous_labels = labels.iloc[contig, :]
        begin = contiguous_labels.iloc[0]["begin spect idx"]
        end = contiguous_labels.iloc[-1]["end spect idx"]
        spect_slice = spect[:, begin:end]
        if end - begin == 0:
            continue

        label_vector = np.zeros((spect_slice.shape[1]))
        first = True

        for _, row in contiguous_labels.iterrows():
            if row["Sound_Type"] in excluded_classes:
                continue
            if first:
                overall_begin = row["begin spect idx"]
                first = False

            sound_begin = row["begin spect idx"] - overall_begin
            sound_end = row["end spect idx"] - overall_begin
            label_vector[sound_begin:sound_end] = name_to_class_code[row["Sound_Type"]]

        features_and_labels.append([spect_slice, label_vector])

    return features_and_labels


def convert_time_to_index(time, sample_rate):
    """
    Converts time (seconds) to .wav file index.
    :param time: int. The time to convert.
    :param sample_rate: int. The sample rate of the .wav file
    :return: int. converted index.
    """
    return np.round(time * sample_rate).astype(np.int)


def process_wav_file(
    wav_filename, csv_filename, n_fft, mel_scale, config, hop_length=200
):
    """
    Applies the labels contained in csv_filename to the .wav file and extracts the labeled regions.
    n_fft controls to number of fast fourier transform components to
    :param wav_filename: .wav file containing the recording.
    :param csv_filename: Csv file containing the labels.
    :param n_fft: Number of ffts to use when calculating the spectrogram.
    :param mel_scale: bool. Whether or not to calculate a MelSpectrogram.
    :param config: beetles.Config object. Controls the mapping from class name to class code and the classes to exclude
    from the labeled sounds.
    :param hop_length: int. Hop length between subsequent fft calculations when forming the spectrogram.
    :return: list: List of lists. Each sublist contains a feature tensor and the associated vector of labels.
    """
    # reads the csv into a pandas df called labels, extracts waveform and sample_rate.
    labels = pd.read_csv(csv_filename)
    waveform, sample_rate = torchaudio.load(wav_filename)
    # adds additional columns to give indices of these chirp locations
    labels["begin idx"] = convert_time_to_index(labels["Begin Time (s)"], sample_rate)
    labels["end idx"] = convert_time_to_index(labels["End Time (s)"], sample_rate)

    # creates a spectrogram with a log2 transform
    if mel_scale:
        spect = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length
        )(waveform)
    else:
        spect = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)(
            waveform
        )

    # dictionary containing all pre-labeled chirps and their associated spectrograms
    spect = spect.squeeze()
    features_and_labels = create_label_to_spectrogram(
        spect,
        labels,
        hop_length=hop_length,
        name_to_class_code=config.name_to_class_code,
        excluded_classes=config.excluded_classes,
    )

    return features_and_labels


def load_csv_and_wav_files_from_directory(data_dir):
    """
    Loads .csv and .wav files from the data directory. Assumes a hierarchical structure:
    data_dir
      label_set_1
         label_1.wav
         label_1.csv
      label_set_2
         label_2.wav
         label_2.csv
    It gets these .wav and .csv pairs. If there are multiple .csv or .wav files in the same subdirectory, this function
    passes over them.
    :param data_dir: string. Parent directory of subdirectories containing the .wav and .csv pairs.
    :return: Dict. Keys are the basename of the .wav file and values are two-element lists, containing the .wav filenane as the first
    element and the .csv filename as the second.
    """
    dirs = os.listdir(data_dir)
    csvs_and_wav = {}
    for d in dirs:
        if os.path.isdir(os.path.join(data_dir, d)):
            labels = glob(os.path.join(data_dir, d, "*.csv"))
            wav = glob(os.path.join(data_dir, d, "*WAV")) + glob(
                os.path.join(data_dir, d, "*wav")
            )
            if len(wav) == 1 and len(labels) == 1:
                csvs_and_wav[os.path.splitext(os.path.basename(wav[0]))[0]] = [
                    wav[0],
                    labels[0],
                ]
            elif len(wav) == 0 or len(labels) == 0:
                print(
                    "found {} wav files and {} csvs in directory {}".format(
                        len(wav), len(labels), d
                    )
                )
            else:
                print(
                    "found {} wav files and {} csvs in directory {}.\n"
                    "Passing.".format(len(wav), len(labels), d)
                )

    return csvs_and_wav


def save_data(out_path, data_list, index_to_label):
    """
    Saves features and labels as pickled numpy arrays.
    Throws away labels > 10000 records. Each pickled array is assigned a filename based on the maximum number of
    class labels in the label vector.
    :param out_path: str. Where to save the pickled data.
    :param data_list: List of 2-element lists, where the first element are the features and the second the point-wise
    label vector.
    :param index_to_label: Mapping from class index to class name.
    :return: None.
    """
    os.makedirs(out_path, exist_ok=True)

    for i, (features, label_vector) in enumerate(data_list):

        if label_vector.shape[0] > 10000:
            # TODO: fix this error in create_label_to_spectrogram
            # way too big - indicates error in label assign
            continue
        uniq = np.unique(label_vector, return_counts=True)
        label = np.argmax(uniq[1])
        if index_to_label[label] == "X" and len(uniq[0]) != 1:
            lvec = label_vector[label_vector != index_to_label["BACKGROUND"]]
            uniq = np.unique(lvec, return_counts=True)
            label = np.argmax(uniq[1])

        out_fpath = os.path.join(
            out_path, index_to_label[label] + "_" + str(i) + ".pkl"
        )

        with open(out_fpath, "wb") as dst:
            pickle.dump([features.numpy(), label_vector], dst)


def extract(config, random_seed, no_mel_scale, n_fft, data_dir, output_data_path):
    """
    Function to wrap the data loading, extraction, and saving routine.
    Loads data from data_dir, calculates spectrogram, extracts labeled regions based on the .csv
    file, and saves the regions to disk after shuffling and splitting into train/test/validation splits randomly.
    :param config: beetles.Config() object.
    :param random_seed: int. What to seed RNGs with for reproducibility.
    :param no_mel_scale: Whether or not to use the mel scale.
    :param n_fft: How many ffts to use when calculating the spectrogram.
    :param data_dir: Where the .wav and .csv files are stored.
    :param output_data_path: Where to save the extracted data.
    :return: None.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    mel = not no_mel_scale

    csv_and_wav = load_csv_and_wav_files_from_directory(data_dir)

    out = []

    for filename, (wav, csv) in csv_and_wav.items():
        features_and_labels = process_wav_file(wav, csv, n_fft, mel, config)
        out.extend(features_and_labels)

    random.shuffle(out)
    indices = np.arange(len(out))
    train_idx, test_idx, _, _ = train_test_split(
        indices, indices, test_size=0.15, random_state=random_seed
    )
    test_idx, val_idx, _, _ = train_test_split(
        test_idx, test_idx, test_size=0.5, random_state=random_seed
    )

    train_split = np.asarray(out)[train_idx]
    val_split = np.asarray(out)[val_idx]
    test_split = np.asarray(out)[test_idx]

    train_path = os.path.join(output_data_path, "train")
    validation_path = os.path.join(output_data_path, "validation")
    test_path = os.path.join(output_data_path, "test")

    save_data(train_path, train_split, config.index_to_label)
    save_data(validation_path, val_split, config.index_to_label)
    save_data(test_path, test_split, config.index_to_label)
