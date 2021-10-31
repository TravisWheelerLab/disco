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

from beetles import INDEX_TO_LABEL, LABEL_TO_INDEX, EXCLUDED_CLASSES


def w2s_idx(idx, hop_length):
    # waveform to spectrogram index
    return idx // hop_length


def create_label_to_spectrogram(spect, labels, hop_length, neighbor_tolerance=100):
    """
    Accepts a spectrogram (torch.Tensor) and labels (pd.DataFrame) and returns
    a song type and a list of tensors of those spectrograms as a value. e.g.:
    len(label_to_spectrogram['A']) = 49 (number of sounds of this subtype)
    type(label_to_spectrogram['A']) = class 'list'
    type(label_to_spectrogram['A'][0]) = class 'torch.Tensor'
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
            if row["Sound_Type"] in EXCLUDED_CLASSES:
                continue
            if first:
                overall_begin = row["begin spect idx"]
                first = False
            # have to do the shifty shift
            sound_begin = row["begin spect idx"] - overall_begin
            sound_end = row["end spect idx"] - overall_begin
            label_vector[sound_begin:sound_end] = LABEL_TO_INDEX[row["Sound_Type"]]
        features_and_labels.append([spect_slice, label_vector])

    return features_and_labels


def convert_time_to_index(time, sample_rate):
    # np.round is good enough for our purposes
    # since we have a really high sample rate, and the chirps exist for a second or two
    return np.round(time * sample_rate).astype(np.int)


def process_wav_file(wav_filename, csv_filename, n_fft, mel_scale, hop_length=200):
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
        spect, labels, hop_length=hop_length
    )

    return features_and_labels


def load_csv_and_wav_files_from_directory(data_dir):
    # takes in a directory String and returns a dictionary with a key as the file label and a value
    # as a list with index 0 as the wav file and index 1 as the csv file
    dirs = os.listdir(data_dir)
    csvs_and_wav = {}
    for d in dirs:
        if os.path.isdir(os.path.join(data_dir, d)):
            labels = glob(os.path.join(data_dir, d, "*.csv"))
            wav = glob(os.path.join(data_dir, d, "*WAV")) + glob(
                os.path.join(data_dir, d, "*wav")
            )
            if len(wav) and len(labels):
                csvs_and_wav[os.path.splitext(os.path.basename(wav[0]))[0]] = [
                    wav[0],
                    labels[0],
                ]
            else:
                # print('found {} wav files and {} csvs in directory {}'.format(len(wav), len(labels), d))
                pass
    return csvs_and_wav


def form_spectrogram_type(mel, n_fft):
    # creates a string to attach to the BeetleFile object that will allow for offloading the files in data_loader
    # to go in the correct directory that matches the type of spectrogram we created.
    directory_location = ""

    # add mel information
    if mel:
        directory_location = directory_location + "mel_"
    else:
        directory_location = directory_location + "no_mel_"

    directory_location = directory_location + str(n_fft)

    return directory_location


def save_data(out_path, data_list):
    os.makedirs(out_path, exist_ok=True)

    for i, (features, label_vector) in enumerate(data_list):

        if label_vector.shape[0] > 10000:
            # TODO: fix this error in create_label_to_spectrogram
            # way too big - indicates error in label assign
            continue
        uniq = np.unique(label_vector, return_counts=True)
        label = np.argmax(uniq[1])
        if INDEX_TO_LABEL[label] == "X" and len(uniq[0]) != 1:
            lvec = label_vector[label_vector != LABEL_TO_INDEX["BACKGROUND"]]
            uniq = np.unique(lvec, return_counts=True)
            label = np.argmax(uniq[1])

        out_fpath = os.path.join(
            out_path, INDEX_TO_LABEL[label] + "_" + str(i) + ".pkl"
        )

        with open(out_fpath, "wb") as dst:
            pickle.dump([features.numpy(), label_vector], dst)


def main(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    mel = True if not args.no_mel_scale else False  # i want default True
    n_fft = args.n_fft

    data_dir = args.data_dir

    csv_and_wav = load_csv_and_wav_files_from_directory(data_dir)

    out = []

    for filename, (wav, csv) in csv_and_wav.items():
        features_and_labels = process_wav_file(wav, csv, n_fft, mel)
        out.extend(features_and_labels)

    random.shuffle(out)
    indices = np.arange(len(out))
    train_idx, test_idx, _, _ = train_test_split(
        indices, indices, test_size=0.15, random_state=args.random_seed
    )
    test_idx, val_idx, _, _ = train_test_split(
        test_idx, test_idx, test_size=0.5, random_state=args.random_seed
    )

    train_split = np.asarray(out)[train_idx]
    val_split = np.asarray(out)[val_idx]
    test_split = np.asarray(out)[test_idx]

    train_path = os.path.join(args.output_data_path, "train")
    validation_path = os.path.join(args.output_data_path, "validation")
    test_path = os.path.join(args.output_data_path, "test")

    save_data(train_path, train_split)
    save_data(validation_path, val_split)
    save_data(test_path, test_split)
