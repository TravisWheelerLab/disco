from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class SpectrogramDataset:
    def __init__(self,
                 directory_name):
        self.directory_name = directory_name
        # spectrograms_list[i][0] is the label, [i][1] is the spect.
        self.spectrograms_list, self.max_spec_length, self.unique_labels = self.load_in_all_files(self.directory_name)

    def load_in_all_files(self, directory_name, filterlabel="Y"):
        max_length = 999999
        spectrograms_list = []
        root = './' + directory_name + "/spect"
        files = glob(os.path.join(root, "*"))
        class_counter = defaultdict(int)
        for filepath in files:
            head, tail = os.path.split(filepath)
            label = tail.split(".")[0]
            spect = np.load(filepath)
            if label != filterlabel:
                class_counter[label] += 1
                max_length = min(max_length, spect.shape[1])
                spectrograms_list.append([label, spect])
        return spectrograms_list, max_length, class_counter

    def __getitem__(self, idx):
        label = self.spectrograms_list[idx][0]
        spect = self.spectrograms_list[idx][1]
        return label, spect

    def __len__(self):
        return len(self.spectrograms_list)

    def generate_bar_chart(self):
        labels = list(self.unique_labels.keys())
        counts = list(self.unique_labels.values())
        plt.bar(labels, counts)
        plt.show()

    def get_unique_labels(self):
        labels = []
        for i in range(len(self.spectrograms_list)):
            if self.spectrograms_list[i][0] not in labels:
                labels.append(self.spectrograms_list[i][0])
        labels.sort()
        return labels


if __name__ == '__main__':
    train_data = SpectrogramDataset("train_data")
    train_data.generate_bar_chart()
    print(len(train_data))

    test_data = SpectrogramDataset("test_data")
    test_data.generate_bar_chart()
    print(len(test_data))
