import numpy as np
import os
import pandas as pd
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torchaudio
import matplotlib.pyplot as plt

from collections import defaultdict
from glob import glob
from sklearn.cluster import KMeans

from data_feeder import LABEL_TO_INDEX, EXCLUDED_CLASSES


class BeetleFile:

    def __init__(self,
                 name,
                 csv_path,
                 wav_path,
                 spectrogram,
                 label_to_spectrogram,
                 mel,
                 n_fft,
                 log,
                 vert_trim):
        self.name = name
        self.csv_path = csv_path
        self.wav_path = wav_path
        self.spectrogram = spectrogram.squeeze()
        self.label_to_spectrogram = label_to_spectrogram
        self.k = None
        self.k_subset = None
        self.spectrogram_type = form_spectrogram_type(mel, n_fft, log, vert_trim)

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

    def plot_histogram(self, keys_to_plot, begin_cutoff_idx=None, end_cutoff_idx=None):
        # plots a histogram that sums over designated vertical band for visualization of the spectrogram
        # and comparison with k-means clustering values
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

    def plot_subclass_histogram(self, keys_to_plot, begin_cutoff_idx=None, end_cutoff_idx=None):
        labeled_data_from_keys = []
        for key in keys_to_plot:
            x = [k[begin_cutoff_idx:end_cutoff_idx] for k in self.label_to_spectrogram[key]]
            labeled_data_from_keys.extend(x)
        a = np.column_stack(labeled_data_from_keys)
        a_and_b = np.sum(a, axis=0)
        plt.hist(a_and_b, bins=50, histtype='step', color='r')
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

    def random_sample_from_entire_spectrogram(self, length_of_sample, vertical_trim):
        # takes a random spot on the spectrogram of a given file with a given length
        center = int(np.random.rand() * self.spectrogram.shape[-1])
        return self.spectrogram[vertical_trim:, center - length_of_sample // 2:center + length_of_sample // 2]


def form_spectrogram_type(mel, n_fft, log, vert_trim):
    # creates a string to attach to the BeetleFile object that will allow for offloading the files in data_loader
    # to go in the correct directory that matches the type of spectrogram we created.
    directory_location = ''

    # add mel information
    if mel:
        directory_location = directory_location + 'mel_'
    else:
        directory_location = directory_location + 'no_mel_'

    # add logged information
    if log:
        directory_location = directory_location + 'log_'
    else:
        directory_location = directory_location + 'no_log_'

    directory_location = directory_location + str(n_fft) + '_'

    # add information about desired vertical cutoff
    if vert_trim == 0:
        directory_location = directory_location + 'no_vert_trim'
    else:
        directory_location = directory_location + 'vert_trim_' + str(vert_trim)

    return directory_location


def load_csv_and_wav_files_from_directory(data_dir):
    # takes in a directory String and returns a dictionary with a key as the file label and a value
    # as a list with index 0 as the wav file and index 1 as the csv file
    dirs = os.listdir(data_dir)
    csvs_and_wav = {}
    for d in dirs:
        if os.path.isdir(os.path.join(data_dir, d)):
            labels = glob(os.path.join(data_dir, d, "*.csv"))
            wav = glob(os.path.join(data_dir, d, "*WAV")) + glob(os.path.join(data_dir, d, "*wav"))
            if len(wav) and len(labels):
                csvs_and_wav[os.path.splitext(os.path.basename(wav[0]))[0]] = [wav[0], labels[0]]
            else:
                # print('found {} wav files and {} csvs in directory {}'.format(len(wav), len(labels), d))
                pass
    return csvs_and_wav


def convert_time_to_index(time, sample_rate):
    # np.round is good enough for our purposes
    # since we have a really high sample rate, and the chirps exist for a second or two
    return np.round(time * sample_rate).astype(np.int)


def w2s_idx(idx, hop_length):
    # waveform to spectrogram index
    return idx // hop_length


def create_label_to_spectrogram(spect, labels, hop_length, vertical_trim,
                                neighbor_tolerance=100):
    # takes in a specific spectrogram dictated by process_wav_file and returns a dictionary with a key as
    # a song type and a list of tensors of those spectrograms as a value. e.g.:
    # len(label_to_spectrogram['A']) = 49 (number of sounds of this subtype)
    # type(label_to_spectrogram['A']) = class 'list'
    # type(label_to_spectrogram['A'][0]) = class 'torch.Tensor'
    labels['begin spect idx'] = [w2s_idx(x, hop_length) for x in labels['begin idx']]
    labels['end spect idx'] = [w2s_idx(x, hop_length) for x in labels['end idx']]

    contiguous_indices = []
    if labels.shape[0] == 1:
        contiguous_indices.append([0])
    else:
        labels = labels.sort_values(by='begin idx')
        i = 0
        while i < labels.shape[0] - 1:
            contig = [i]
            while (labels.iloc[i + 1]['begin idx'] - labels.iloc[i]['end idx']) <= neighbor_tolerance:
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
        begin = contiguous_labels.iloc[0]['begin spect idx']
        end = contiguous_labels.iloc[-1]['end spect idx']
        spect_slice = spect[:, begin:end]
        if end-begin == 0:
            continue
        end = end - begin # label indices must be relative to the beginning
        # of the array
        begin = 0
        label_vector = np.zeros((spect_slice.shape[1]))
        first = True
        for _, row in contiguous_labels.iterrows():
            if row['Sound_Type'] in EXCLUDED_CLASSES:
                continue
            if first:
                overall_begin = row['begin spect idx']
                first = False
            # have to do the shifty shift
            sound_begin = row['begin spect idx'] - overall_begin
            sound_end = row['end spect idx'] - overall_begin
            label_vector[sound_begin:sound_end] = LABEL_TO_INDEX[row['Sound_Type']]
        features_and_labels.append([spect_slice, label_vector])

    return features_and_labels


def process_wav_file(wav_filename, csv_filename, n_fft, mel_scale, log_spectrogram, vertical_trim=None):
    # reads the csv into a pandas df called labels, extracts waveform and sample_rate.
    labels = pd.read_csv(csv_filename)
    waveform, sample_rate = torchaudio.load(wav_filename)
    # adds additional columns to give indices of these chirp locations
    labels['begin idx'] = convert_time_to_index(labels['Begin Time (s)'], sample_rate)
    labels['end idx'] = convert_time_to_index(labels['End Time (s)'], sample_rate)

    # creates a spectrogram with a log2 transform
    hop_length = 200
    if mel_scale:
        spect = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length)(waveform)
        if log_spectrogram:
            spect = spect.log2()
    else:
        spect = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)(waveform)
        if log_spectrogram:
            spect = spect.log2()

    # dictionary containing all pre-labeled chirps and their associated spectrograms
    spect = spect.squeeze()
    features_and_labels = create_label_to_spectrogram(spect, labels, hop_length=hop_length,
                                                      vertical_trim=vertical_trim)

    return features_and_labels


def fit_kmeans(spect, begin_cutoff_idx, end_cutoff_idx, n_clusters=2):
    cutoff_spectrogram = spect[begin_cutoff_idx:end_cutoff_idx]
    k = KMeans(n_clusters=n_clusters).fit(np.sum(cutoff_spectrogram, axis=0).reshape(-1, 1))
    return k


def determine_default_vert_trim(mel, log, n_fft):
    vertical_index = 0
    if mel:
        if log:
            if n_fft >= 1400:
                vertical_index = 15
            elif n_fft >= 600:
                vertical_index = 20
            else:
                vertical_index = 35
        # unsure about what contributes best to accuracy if it is unlogged.
    else:
        if log:
            vertical_index = 5
    return vertical_index


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mel_scale', action='store_true')
    ap.add_argument('--log_scale', required=True, action='store_true')
    ap.add_argument('--n_fft', required=True, type=int)
    ap.add_argument('--vert_trim', required=False, default=None)
    ap.add_argument('--save_fig', required=True, action='store_false')
    return ap.parse_args()


if __name__ == '__main__':

    args = parser()
    # mel = args.mel_scale
    # log = args.log_scale
    # n_fft = args.n_fft
    # vert_trim = args.vert_trim
    # savefig = args.save_fig

    mel = True
    log = True
    n_fft = 800
    vert_trim = 30
    savefig = False

    if vert_trim is None:
        vert_trim = determine_default_vert_trim(mel, log, n_fft)
        print('vert trim has been set to', vert_trim)

    data_dir = './wav-files-and-annotations/'
    csvs_and_wav = load_csv_and_wav_files_from_directory(data_dir)

    beetle_files = {}

    i = 0
    plt.style.use("dark_background")
    for filename, (wav, csv) in csvs_and_wav.items():
        if i == 3:
            spectrogram, label_to_spectrogram = process_wav_file(wav, csv, n_fft, mel, log, vert_trim)
            beetle_files[filename] = BeetleFile(filename, csv, wav, spectrogram, label_to_spectrogram, mel, n_fft, log,
                                                vert_trim)
            bf = beetle_files[filename]
            sounds_list = ['A', 'B']
            for j in range(3):
                for sound_type in sounds_list:
                    if sound_type in bf.label_to_spectrogram.keys():
                        if not log:
                            plt.imshow(bf.label_to_spectrogram[sound_type][j].log2())
                        else:
                            plt.imshow(bf.label_to_spectrogram[sound_type][j])
                        title = sound_type + str(j)
                        plt.title(title)
                        plt.colorbar()
                        plt.show()
                        if savefig:
                            plt.savefig('image_offload/' + title + bf.spectrogram_type + '.png')
                            print(title, "saved.")
                        plt.close()
        i += 1
