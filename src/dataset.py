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
                 data_path,
                 spect_type,
                 max_spec_length=40,
                 filtered_sounds=['C', 'Y'],
                 clip_spects=True,
                 bootstrap_sample=False):

        self.spect_lengths = defaultdict(list)
        self.dataset_type = dataset_type
        self.data_path = data_path
        self.spect_type = spect_type
        self.max_spec_length = max_spec_length
        self.filtered_sounds = filtered_sounds
        self.clip_spects = clip_spects
        self.bootstrap_sample = bootstrap_sample
        # spectrograms_list[i][0] is the label, [i][1] is the spect.
        self.spectrograms_list, self.unique_labels = self.load_in_all_files(self.dataset_type, self.spect_type,
                                                                            self.filtered_sounds, self.bootstrap_sample)

    def load_in_all_files(self, dataset_type, spect_type, filtered_labels, bootstrap_sample):

        spectrograms_list = []
        root = os.path.join(self.data_path, dataset_type, spect_type, 'spect')
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

        if bootstrap_sample:
            indices = np.random.choice(len(spectrograms_list), size=len(spectrograms_list), replace=True)
            bootstrapped_spects = []
            for i in range(indices.shape[-1]):
                spect_to_add_label = spectrograms_list[indices[i]][0]
                spect_to_add = spectrograms_list[indices[i]][1]
                bootstrapped_spects.append([spect_to_add_label, spect_to_add])
            spectrograms_list = bootstrapped_spects

        return spectrograms_list, sorted_class_counter

    def __getitem__(self, idx):
        # returns a tuple with [0] the class label and [1] a slice of the spectrogram or the entire image.
        label = self.spectrograms_list[idx][0]
        spect = self.spectrograms_list[idx][1]

        num_col = spect.shape[1]
        spect_size = self.max_spec_length

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

    def get_unique_labels(self):
        return self.unique_labels.keys()
