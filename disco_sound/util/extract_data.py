import logging
import os
import pickle
import random
from glob import glob
from shutil import copy2

import numpy as np
import pandas as pd
import torchaudio

from disco_sound.util.util import add_gaussian_beeps, add_white_noise

logger = logging.getLogger(__name__)


def w2s_idx(idx, hop_length):
    """
    Converts .wav file index to spectrogram index.
    :param idx: Index to convert.
    :param hop_length: Length between subsequent spectrogram windows.
    :return: Converted index.
    """
    return idx // hop_length


def create_label_to_spectrogram(
    spect,
    labels,
    hop_length,
    name_to_class_code,
    excluded_classes,
    neighbor_tolerance=100,
    extract_context=None,
):
    """
    Accepts a spectrogram (torch.Tensor) and labels (pd.DataFrame) and returns
    a song type and a list of tensors of those spectrograms as a value. e.g.:

    len(label_to_spectrogram['A']) = 49 (number of sounds of this subtype)
    type(label_to_spectrogram['A']) = class 'list'
    type(label_to_spectrogram['A'][0]) = class 'torch.Tensor'

    :param spect: torch.Tensor of the entire spectrogram of the sound file.
    :param labels: pandas dataframe of raven-format labels of corresponding spectrogram.
    :param hop_length: int. Length between subsequent spectrogram windows.
    :param name_to_class_code: dictionary. In config, maps label to its class index.
    :param excluded_classes: tuple. Any classes left out of training that may be on the Raven .csvs.
    :param neighbor_tolerance: int. If two labeled regions are within :param: neighbor_tolerance of one another they'll
    be considered one example and saved as one array.
    :return: list: List of lists. Each sublist contains a feature tensor and the corresponding vector of labels.
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

        features_and_labels.append([spect_slice.numpy(), label_vector])

    return features_and_labels


def convert_time_to_index(time, sample_rate):
    """
    Converts time (seconds) to .wav file index.
    :param time: int. The time to convert to an index.
    :param sample_rate: int. The sample rate of the .wav file.
    :return: int. converted index.
    """
    return np.round(time * sample_rate).astype(np.int)


def extract_wav_and_csv_pair(
    csv_filename,
    wav_filename,
    n_fft,
    mel_scale,
    snr,
    add_beeps,
    hop_length=200,
    name_to_class_code=None,
    excluded_classes=None,
    extract_context=None,
):

    labels = pd.read_csv(csv_filename)
    logger.info(f"File {os.path.basename(csv_filename)} has {labels.shape} labels.")

    if not os.path.isfile(wav_filename):
        wav_filename = os.path.splitext(csv_filename)[0] + ".WAV"
        if not os.path.isfile(wav_filename):
            raise ValueError(
                f"couldn't find .wav file at {wav_filename}. "
                f"Make sure the correct .wav file is in the filename column of {csv_filename},"
                f" or that the .wav file has the name filename as the csv but with the .wav extension."
            )

    waveform, sample_rate = torchaudio.load(wav_filename)

    if snr > 0:
        waveform = add_white_noise(waveform, snr=snr)

    if add_beeps:
        waveform = add_gaussian_beeps(waveform, sample_rate=sample_rate)

    # adds additional columns to give indices of these chirp locations
    labels["begin idx"] = convert_time_to_index(labels["Begin Time (s)"], sample_rate)
    labels["end idx"] = convert_time_to_index(labels["End Time (s)"], sample_rate)

    # creates a spectrogram with a log2 transform
    if mel_scale:
        spect = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length
        )(waveform)
    else:
        spect = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
        )(waveform)

    # dictionary containing all pre-labeled chirps and their associated spectrograms
    spect = spect.squeeze()

    features_and_labels = create_label_to_spectrogram(
        spect,
        labels,
        hop_length=hop_length,
        name_to_class_code=name_to_class_code,
        excluded_classes=excluded_classes,
        extract_context=extract_context,
    )

    return features_and_labels


def save_data(out_path, data_list, filename_prefix, index_to_label, overwrite):
    """
    Saves features and labels as pickled numpy arrays.
    Throws away labels > 10000 records. Each pickled array is assigned a filename based on the maximum number of
    class labels in the label vector.
    :param out_path: str. Where to save the pickled data.
    :param overwrite: str. Where to save the pickled data.
    :param filename_prefix: str. What to prefix the pickle files with.
    :param data_list: List of 2-element lists, where the first element are the features and the second the point-wise
    label vector.
    :param index_to_label: Mapping from class index to class name.
    :return: None.
    """
    os.makedirs(out_path, exist_ok=True)
    fcount = 0

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
            out_path, f"{filename_prefix}_{index_to_label[label]}_{fcount}.pkl"
        )

        if os.path.isfile(out_fpath) and not overwrite:
            logger.info(
                f"Found file already at {out_fpath}. Skipping re-saving."
                f" Specify --overwrite to overwrite."
            )
        else:
            logger.info(f"Saving {out_fpath}.")
            with open(out_fpath, "wb") as dst:
                pickle.dump([features, label_vector], dst)

        fcount += 1


def extract_single_file(
    csv_file,
    wav_file,
    seed,
    no_mel_scale,
    n_fft,
    output_data_path,
    overwrite,
    snr,
    add_beeps,
    class_code_to_name,
    name_to_class_code,
    excluded_classes,
    extract_context=None,
):
    """
    Extract data from a single .wav and .csv pair.
    """
    logger.info(f"Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)

    mel = not no_mel_scale
    features_and_labels = extract_wav_and_csv_pair(
        csv_file,
        wav_file,
        n_fft,
        mel,
        snr,
        add_beeps,
        name_to_class_code=name_to_class_code,
        excluded_classes=excluded_classes,
        extract_context=extract_context,
    )
    save_data(
        output_data_path,
        features_and_labels,
        filename_prefix=os.path.basename(os.path.splitext(csv_file)[0]),
        index_to_label=class_code_to_name,
        overwrite=overwrite,
    )


def shuffle_data(data_directory, train_pct, extension, move, seed):
    data_files = glob(os.path.join(data_directory, f"*{extension}"))

    logger.info(f"Setting seed for shuffling to {seed}.")

    np.random.seed(seed)
    indices = np.random.permutation(len(data_files))
    train_idx = indices[: int(len(indices) * train_pct)]
    the_rest = indices[int(len(indices) * train_pct) :]
    test_idx = the_rest[: len(the_rest) // 2]
    val_idx = the_rest[len(the_rest) // 2 :]
    assert np.all(train_idx != test_idx)
    assert np.all(train_idx != val_idx)
    assert np.all(test_idx != val_idx)

    train_split = [data_files[idx] for idx in train_idx]
    test_split = [data_files[idx] for idx in test_idx]
    val_split = [data_files[idx] for idx in val_idx]

    train_path = os.path.join(data_directory, "train")

    validation_path = os.path.join(data_directory, "validation")
    test_path = os.path.join(data_directory, "test")

    logger.info(f"Saving train files to {train_path}.")
    copy_or_move_files(train_path, train_split, move)
    logger.info(f"Saving validation files to {validation_path}.")
    copy_or_move_files(validation_path, val_split, move)
    logger.info(f"Saving test files to {test_path}.")
    copy_or_move_files(test_path, test_split, move)


def copy_or_move_files(out_path, files, move):
    os.makedirs(out_path, exist_ok=True)
    # instead of a conditional
    func = os.rename if move else copy2
    for filename in files:
        logger.debug(f"Copying {filename} to {out_path}")
        func(filename, os.path.join(out_path, os.path.basename(filename)))
