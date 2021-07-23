from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, OrderedDict
import random
import torch
import pdb

LABEL_TO_INDEX = {'A': 0, 'B': 1, 'X': 2}


class SpectrogramDataset(torch.utils.data.Dataset):

    def __init__(self,
                 directory_name,
                 max_spec_length=40,
                 filtered_sounds=['C', 'Y'],
                 clip_spects=True):
        self.directory_name = directory_name
        self.max_spec_length = max_spec_length
        self.filtered_sounds = filtered_sounds
        self.clip_spects = clip_spects
        # spectrograms_list[i][0] is the label, [i][1] is the spect.
        self.spectrograms_list, self.unique_labels = self.load_in_all_files(self.directory_name, self.filtered_sounds)

    def load_in_all_files(self, directory_name, filtered_labels):
        # saves spectrograms into a list and creates a sorted dictionary (sorted_class_counter) of the counts of each
        # sound class, which is mainly meant for creating the bar chart. also ensures filtering of unwanted labels.
        spectrograms_list = []
        root = './' + directory_name + '/spect'
        # root = os.path.join(directory_name, 'spect')
        files = glob(os.path.join(root, "*"))
        class_counter = defaultdict(int)
        for filepath in files:
            head, tail = os.path.split(filepath)
            label = tail.split(".")[0]
            spect = np.load(filepath)
            if label not in filtered_labels and spect.shape[1] >= self.max_spec_length:
                class_counter[label] += 1
                spectrograms_list.append([label, spect])
        sorted_class_counter = OrderedDict(sorted(class_counter.items()))
        return spectrograms_list, sorted_class_counter

    def __getitem__(self, idx):
        # returns a tuple with [0] the class label and [1] the spectrogram slice
        # which is based on self.max_spec_length.
        label = self.spectrograms_list[idx][0]
        spect = self.spectrograms_list[idx][1]
        num_col = spect.shape[1]
        random_index = round(random.uniform(0, num_col - self.max_spec_length))
        if self.clip_spects:
            spect_slice = torch.tensor(spect[:, random_index:random_index + self.max_spec_length]).unsqueeze(0)
            label_tensor = torch.tensor(np.repeat(a=LABEL_TO_INDEX[label], repeats=self.max_spec_length))
        else:
            spect_slice = torch.tensor(spect).unsqueeze(0)
            label_tensor = torch.tensor(np.repeat(a=LABEL_TO_INDEX[label], repeats=len(spect[1])))
        return spect_slice, label_tensor

    def __len__(self):
        return len(self.spectrograms_list)

    def generate_bar_chart(self):
        # saves bar chart created from unique_labels dictionary into 'image_offload' directory
        labels = list(self.unique_labels.keys())
        counts = list(self.unique_labels.values())
        plt.bar(labels, counts)
        plt.savefig('image_offload/' + "bar_chart_" + self.directory_name + '.png')

    def get_unique_labels(self):
        return self.unique_labels.keys()


if __name__ == '__main__':
    train_data = SpectrogramDataset(directory_name="train_data", clip_spects=False)
    train_data.generate_bar_chart()
    print(len(train_data))
    print(train_data[25])
    print('spect size:', train_data[25][0].shape)
