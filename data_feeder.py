from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np


class SpectrogramDataset:
    def __init__(self,
                 directory_name):  # String - "train_data" or "test_data"

        self.directory_name = directory_name
        # spectrograms_list[0] is the label, [1] is the spect.
        self.spectrograms_list, self.max_spec_length = self.load_in_all_files(self.directory_name)

    def load_in_all_files(self, directory_name):
        max_length = 999999
        spectrograms_list = []
        root = './' + directory_name + "/spect"
        files = glob(os.path.join(root, "*"))
        i = 0
        j = 0
        class_counter = {"A": 0, "B": 0, "C": 0, "X":0, "Y":0}
        running_length_sum = 0
        for filepath in files:
            head, tail = os.path.split(filepath)
            label = tail.split(".")[0]
            spect = np.load(filepath)
            minval = 50
            running_length_sum += spect.shape[1]
            if spect.shape[1] < max_length:
                print(spect.shape[1], head + "/" + tail)
            if spect.shape[1] < minval:
                i += 1
                if label == "A" or "B" or "C" or "X" or "Y":
                    class_counter[label] += 1
                else:
                    print(label)
            max_length = min(max_length, spect.shape[1])
            spectrograms_list.append([label, spect])
            j += 1
        print(i, "are smaller than", minval, "columns out of", j)
        print(class_counter)
        print ("average size of all:", running_length_sum/len(spectrograms_list))
        return spectrograms_list, max_length

    def __getitem__(self, index, load_all=True):
        pass
        # s = get_spectrogram_from_list()
        # sub_s = get_random_contiguous_sample_from_spect(s)
        # return sub_s, label_of_s

    def __len__(self):
        return len(self.sound_labels)

    def generate_bar_chart(self):
        labels_dict = dict.fromkeys(self.get_unique_labels(), 0)
        for spect in self.spectrograms_list:
            labels_dict[spect[0]] += 1
        labels = list(labels_dict.keys())
        counts = list(labels_dict.values())
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
    my_tester = SpectrogramDataset("train_data")
    my_tester.generate_bar_chart()
