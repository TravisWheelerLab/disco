import os
import random
from collections import defaultdict, OrderedDict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch

import spectrogram_analysis as sa

LABEL_TO_INDEX = {'A': 0, 'B': 1, 'X': 2}
INDEX_TO_LABEL = {0: 'A', 1: 'B', 2: 'X'}


class SpectrogramDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset_type,
                 spect_type,
                 batch_size,
                 max_spec_length=40,
                 filtered_sounds=['C', 'Y'],
                 bin_spects=False,
                 clip_spects=True,
                 n_bins=10):

        self.spect_lengths = defaultdict(list)
        self.dataset_type = dataset_type
        self.spect_type = spect_type
        self.batch_size = batch_size
        self.max_spec_length = max_spec_length
        self.filtered_sounds = filtered_sounds
        self.bin_spects = bin_spects
        self.clip_spects = clip_spects
        self.n_bins = n_bins
        # spectrograms_list[i][0] is the label, [i][1] is the spect.
        self.spectrograms_list, self.unique_labels = self.load_in_all_files(self.dataset_type, self.spect_type,
                                                                            self.filtered_sounds)
        if self.bin_spects:

            self.clip_spects = True
            self.bin_edges, self.bin_index_to_spectogram_index = self.bin_spectrograms(self.spectrograms_list,
                                                                                       n_bins=n_bins)
            self.n_bins = len(self.bin_index_to_spectogram_index)

            self.global_iter = 0

    def bin_spectrograms(self, list_of_spectrograms, n_bins):
        # sort spectrograms by length, then record lower bin edges?
        # TODO: specify bins that are denser in the lower end of the lengths
        log_lengths = np.log10([x[1].shape[-1] for x in list_of_spectrograms])
        hist, bin_edges = np.histogram(log_lengths, bins=n_bins)
        bin_edges = 10**bin_edges
        bin_edges = np.round(bin_edges)
        bin_index_to_spectrogram_index = defaultdict(list)

        for idx, spect in enumerate(list_of_spectrograms):

            num_records = spect[1].shape[-1]
            proper_bin = np.where(bin_edges >= num_records)[0][0]
            for bin_index in range(proper_bin):
                bin_index_to_spectrogram_index[bin_index].append(idx)

        # for looking @ number of members in each (raw) bin
        # print(x)
        # remove bins with less than batch_size number of members for ease
        bin_index_to_spectrogram_index_ = {}
        for bin_index, spectrogram_indices in bin_index_to_spectrogram_index.items():
            if len(spectrogram_indices) >= self.batch_size:
                bin_index_to_spectrogram_index_[bin_index] = spectrogram_indices

        bin_index_to_spectrogram_index = bin_index_to_spectrogram_index_

        # code to debug errors on batch
        # for looking @ number of members in each bin
        # x = list(map(len, bin_index_to_spectrogram_index.values()))
        print('different length bins:', bin_edges[:len(bin_index_to_spectrogram_index)])
        # print(x)

        return bin_edges, bin_index_to_spectrogram_index

    def load_in_all_files(self, dataset_type, spect_type, filtered_labels):
        # saves spectrograms into a list and creates a sorted dictionary (sorted_class_counter) of the counts of each
        # sound class, which is mainly meant for creating the bar chart. also ensures filtering of unwanted labels.

        spectrograms_list = []
        root = os.path.join('data', dataset_type, spect_type, 'spect')
        files = glob(os.path.join(root, "*"))
        class_counter = defaultdict(int)
        for filepath in files:
            head, tail = os.path.split(filepath)
            label = tail.split(".")[0]
            spect = np.load(filepath)
            if label not in filtered_labels and spect.shape[1] >= self.max_spec_length:
                class_counter[label] += 1
                spectrograms_list.append([label, spect])
            self.spect_lengths[label].append(spect.shape[1])
        sorted_class_counter = OrderedDict(sorted(class_counter.items()))
        return spectrograms_list, sorted_class_counter

    def __getitem__(self, idx):
        # returns a tuple with [0] the class label and [1] a slice of the spectrogram or the entire image.
        if self.bin_spects:
            # sample from the index array:
            # could implement some frequency-based sampling on the bins
            # currently there's a uniform distribution on which bins are sampled
            if self.global_iter % self.batch_size == 0:
                self.stateful_random_bin_choice = int(np.random.uniform()*self.n_bins)

            n_bin_members = len(self.bin_index_to_spectogram_index[self.stateful_random_bin_choice])
            random_idx = int(np.random.uniform()*n_bin_members)
            random_spect_idx_from_bin = self.bin_index_to_spectogram_index[self.stateful_random_bin_choice][random_idx]
            label = self.spectrograms_list[random_spect_idx_from_bin][0]
            spect = self.spectrograms_list[random_spect_idx_from_bin][1]
            lower_bin_edge = self.bin_edges[self.stateful_random_bin_choice]
            self.global_iter += 1

        else:
            label = self.spectrograms_list[idx][0]
            spect = self.spectrograms_list[idx][1]

        num_col = spect.shape[1]
        spect_size = int(np.floor(lower_bin_edge)) if self.bin_spects else self.max_spec_length

        random_index = round(random.uniform(0, num_col - spect_size))

        if self.clip_spects:
            spect_slice = torch.tensor(spect[:, random_index:random_index + spect_size])
            label_tensor = torch.tensor(np.repeat(a=LABEL_TO_INDEX[label], repeats=spect_size))
        else:
            spect_slice = torch.tensor(spect)
            label_tensor = torch.tensor(np.repeat(a=LABEL_TO_INDEX[label], repeats=len(spect[1])))
        return spect_slice, label_tensor

    def __len__(self):
        return len(self.spectrograms_list)

    def generate_bar_chart(self):
        # saves bar chart created from unique_labels dictionary into 'image_offload' directory.
        labels = list(self.unique_labels.keys())
        counts = list(self.unique_labels.values())
        plt.style.use("dark_background")
        plt.bar(labels, counts, color='hotpink')
        plt.title('Bar chart of counts of each class in ' + self.dataset_type)
        plt.savefig('image_offload/' + "bar_chart_" + self.dataset_type + '.png')
        plt.close()


    def generate_lengths_histograms(self, plotted_sound_types=['A','B','X'], plot_all=True):
        # saves histograms of lengths for each plotted_sound_type or every labeled sound into 'image_offload' directory.
        if not (plotted_sound_types or plot_all):
            raise ValueError('No plot requirements given. Designate specific sound types or to plot all types.')

        for sound_type in plotted_sound_types:
            plt.style.use("dark_background")
            plt.hist(self.spect_lengths[sound_type], bins=25, color='lightskyblue')
            plt.title('lengths histogram of ' + sound_type + ' in ' + self.dataset_type)
            plt.show()
            file_title = 'lengths_histogram_' + sound_type + '_' + self.dataset_type + '.png'
            plt.savefig('image_offload/' + file_title)
            print('saved ' + file_title + '.')
            plt.close()

        if plot_all:
            all_lengths = []
            for sound_type, lengths_list in self.spect_lengths.items():
                for length in lengths_list:
                    all_lengths.append(length)
            plt.style.use("dark_background")
            plt.hist(all_lengths, bins=25, color='aquamarine')
            plt.title('histogram, all lengths in ' + self.dataset_type)
            plt.show()
            file_title = 'lengths_histogram_' + 'all_lengths_' + self.dataset_type + '.png'
            plt.savefig('image_offload/' + file_title)
            print('saved ' + file_title)
            plt.close()

    def get_unique_labels(self):
        return self.unique_labels.keys()


if __name__ == '__main__':
    mel = True
    log = True
    n_fft = 1600
    vert_trim = None

    if vert_trim is None:
        vert_trim = sa.determine_default_vert_trim(mel, log, n_fft)

    spect_type = sa.form_spectrogram_type(mel, n_fft, log, vert_trim)

    for batch_size in range(1, 256, 32):

        train_data = SpectrogramDataset(dataset_type="train",
                                        spect_type=spect_type,
                                        batch_size=batch_size,
                                        clip_spects=False,
                                        bin_spects=True)

        dataset = torch.utils.data.DataLoader(train_data,
                                              batch_size=batch_size,
                                              drop_last=True)

        for _ in range(10):
            for d in dataset:
                # print(d[0].shape)
                pass