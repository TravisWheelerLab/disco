import numpy as np
import os
import pandas as pd

import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.colors

from collections import defaultdict
from glob import glob
from sklearn.cluster import KMeans

class BeetleFile:

    def __init__(self,
                 name,
                 csv_path,
                 wav_path,
                 spectrogram,
                 label_to_spectrogram):

        self.name = name
        self.csv_path = csv_path
        self.wav_path = wav_path
        self.spectrogram = spectrogram.squeeze()
        self.label_to_spectrogram = label_to_spectrogram
        self.k = None
        self.k_subset = None

    def __str__(self):
        # returns filename for the beetle object
        return self.name

    def __repr__(self):
        return str(self)

    def __getitem__(self, key):
        # used for dictionary lookup
        return self.label_to_spectrogram[key]

    def fit_kmeans(self, n_clusters, begin_cutoff_idx, end_cutoff_idx):
        # finds k-means clusters over entire spectrogram with given vertical slice
        # primary use will be to clip A and B chirps for data pipeline
        self.k = fit_kmeans(self.spectrogram, begin_cutoff_idx, end_cutoff_idx, n_clusters=n_clusters)
        self.k.cluster_centers_ = np.sort(self.k.cluster_centers_, axis=0)

    def fit_kmeans_subset(self, keys_to_fit, n_clusters, begin_cutoff_idx, end_cutoff_idx):
        # same as fit_kmeans, but only grabs from specific labeled data
        labeled_data_from_keys = []
        for key in keys_to_fit:
            x = list(self.label_to_spectrogram[key])
            labeled_data_from_keys.extend(x)
        a = np.column_stack(labeled_data_from_keys)
        self.k_subset = fit_kmeans(a, begin_cutoff_idx, end_cutoff_idx, n_clusters=n_clusters)
        self.k_subset.cluster_centers_ = np.sort(self.k_subset.cluster_centers_, axis=0)

    # plots a histogram that sums over designated vertical band for visualization of the spectrogram
    # and comparison with k-means clustering values
    def plot_histogram(self, keys_to_plot, begin_cutoff_idx=None, end_cutoff_idx=None):

        if begin_cutoff_idx is not None:
            spect = self.spectrogram[begin_cutoff_idx:end_cutoff_idx]
        else:
            spect = self.spectrogram

        # Creates first histogram, sum over designated keys (pre-labeled chirp snippets)
        fig, ax = plt.subplots(ncols=1, figsize=(13, 10))
        ax.hist(np.sum(spect, axis=0), bins=100, histtype='step', label='all values')
        ax1 = ax.twinx()  # Create a twin axes sharing the xaxis.
        labeled_data_from_keys = []
        for key in keys_to_plot:
            x = [k[begin_cutoff_idx:end_cutoff_idx] for k in self.label_to_spectrogram[key]]
            labeled_data_from_keys.extend(x)
        a = np.column_stack(labeled_data_from_keys)

        # Second histogram, sum over all spectrogram and overlay with first
        ax1.hist(np.sum(a, axis=0), bins=75, histtype='step', color='r', label=' and '.join(keys_to_plot))
        ax1.legend()
        ax.legend(loc='upper left')
        plt.show()

    def classify(self, spectrogram, begin_cutoff_idx, end_cutoff_idx):
        # uses k to predict points on a spectrogram based on the column's sum
        if isinstance(spectrogram, torch.Tensor):
            return self.k.predict(spectrogram.numpy()[begin_cutoff_idx:end_cutoff_idx].sum(axis=0).reshape(-1, 1))
        else:
            return self.k.predict(spectrogram[begin_cutoff_idx:end_cutoff_idx].sum(axis=0).reshape(-1, 1))

    def classify_subset(self, spectrogram, begin_cutoff_idx, end_cutoff_idx):
        # same as the previous function but uses k_subset based on given keys
        if isinstance(spectrogram, torch.Tensor):
            return self.k_subset.predict(
                spectrogram.numpy()[begin_cutoff_idx:end_cutoff_idx].sum(axis=0).reshape(-1, 1))
        else:
            return self.k_subset.predict(spectrogram[begin_cutoff_idx:end_cutoff_idx].sum(axis=0).reshape(-1, 1))

    def random_sample_from_class(self, class_key):
        # finds a random example of a pre-labeled chirp with given class key for analysis purposes
        samp = self.label_to_spectrogram[class_key]
        random_idx = int(np.random.rand() * len(samp))
        return samp[random_idx]

    def random_sample_from_entire_spectrogram(self, length_of_sample):
        # takes a random spot on the spectrogram of a given file with a given length
        center = int(np.random.rand() * self.spectrogram.shape[-1])
        return self.spectrogram[:, center - length_of_sample // 2:center + length_of_sample // 2]


def load_csv_and_wav_files_from_directory(data_dir):
    # takes in a directory String and returns a dictionary with a key as the file label and a value
    # as a list with index 0 as the wav file and index 1 as the csv file
    dirs = os.listdir(data_dir)
    csvs_and_wav = {}
    for d in dirs:
        if os.path.isdir(os.path.join(data_dir, d)):
            labels = glob(os.path.join(data_dir, d, "*.csv"))
            wav = glob(os.path.join(data_dir, d, "*WAV")) + glob(os.path.join(data_dir, d, "*wav"))
            if len(wav):
                csvs_and_wav[os.path.splitext(os.path.basename(wav[0]))[0]] = [wav[0], labels[0]]
    return csvs_and_wav


def convert_time_to_index(time, sample_rate):
    # np.round is good enough for our purposes
    # since we have a really high sample rate, and the chirps exist for a second or two
    return np.round(time * sample_rate).astype(np.int)


def w2s_idx(idx, hop_length):
    # waveform to spectrogram index
    return idx // hop_length


def create_label_to_spectrogram(spect, labels):
    # takes in a specific spectrogram dictated by process_wav_file and returns a dictionary with a key as
    # a song type and a list of tensors of those spectrograms as a value. e.g.:
    # len(label_to_spectrogram['A']) = 49 (number of sounds of this subtype)
    # type(label_to_spectrogram['A']) = class 'list'
    # type(label_to_spectrogram['A'][0]) = class 'torch.Tensor'

    # a dictionary typically throws a KeyError if you try to get an item with a key that is not in the dictionary.
    # The defaultdict in contrast will simply create any items that you try to access (provided they do not exist yet).
    label_to_spectrograms = defaultdict(list)
    for _, row in labels.iterrows():
        bi = w2s_idx(int(row['begin idx']), hop_length=200)  # waveform index to spectrogram index conversion
        ei = w2s_idx(int(row['end idx']), hop_length=200)
        label_to_spectrograms[row['Sound_Type']].append(spect[0, :, bi:ei])

    return label_to_spectrograms


def process_wav_file(wav_filename, csv_filename):
    # reads the csv into a pandas object called labels, extracts waveform and sample_rate and saves those.
    labels = pd.read_csv(csv_filename)
    waveform, sample_rate = torchaudio.load(wav_filename)
    # adds additional columns to give indices of these chirp locations
    labels['begin idx'] = convert_time_to_index(labels['Begin Time (s)'], sample_rate)
    labels['end idx'] = convert_time_to_index(labels['End Time (s)'], sample_rate)

    # creates a spectrogram with a log2 transform
    spect = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform).log2()

    # dictionary containing all pre-labeled chirps and their associated spectrograms
    label_to_spectrograms = create_label_to_spectrogram(spect, labels)

    return spect.numpy(), label_to_spectrograms


def fit_kmeans(spect, begin_cutoff_idx, end_cutoff_idx, n_clusters=2):
    cutoff_spectrogram = spect[begin_cutoff_idx:end_cutoff_idx]
    k = KMeans(n_clusters=n_clusters).fit(np.sum(cutoff_spectrogram, axis=0).reshape(-1, 1))
    return k


if __name__ == '__main__':
    CLASS_KEYS = ['X', 'Y', 'A', 'B']
    BINARY_CMAP = matplotlib.colors.ListedColormap(['blue', 'red'])

    data_dir = './wav-files-and-annotations/'
    csvs_and_wav = load_csv_and_wav_files_from_directory(data_dir)

    beetle_files = {}

    for filename, (wav, csv) in csvs_and_wav.items():
        spectrogram, label_to_spectrogram = process_wav_file(wav, csv)
        beetle_files[filename] = BeetleFile(filename, csv, wav, spectrogram, label_to_spectrogram)
    print(beetle_files.keys())
    chosen_file = '2_M12F31_8_24'
    bf = beetle_files[chosen_file] # creates beetle file object
    bci = 35  # begin cutoff index
    eci = 45  # end cutoff index

    bf.fit_kmeans_subset(['A', 'B'], 2, bci, eci)  # fit 2 kmeans on slice of data, using A and B
    print(bf.k_subset.cluster_centers_)
