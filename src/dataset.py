import os
import random
import pickle
from collections import defaultdict, OrderedDict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch

import spectrogram_analysis as sa

LABEL_TO_INDEX = {'A': 0, 'B': 1, 'X': 2}
INDEX_TO_LABEL = {0: 'A', 1: 'B', 2: 'X'}
MASK_FLAG = -1


def pad_batch(batch):

    features = [b[0] for b in batch]
    labels = [b[1] for b in batch]

    mxlen = np.max([f.shape[-1] for f in features])
    padded_batch = torch.zeros((len(batch), features[0].shape[0], mxlen))
    masks = torch.zeros((len(batch), features[0].shape[0], mxlen))
    padded_labels = torch.zeros((len(batch), mxlen)) + MASK_FLAG

    for i, (f, l) in enumerate(zip(features, labels)):
        padded_batch[i, :, :f.shape[-1]] = f
        masks[i, :, f.shape[-1]:] = True
        padded_labels[i, :l.shape[-1]] = l

    return padded_batch, masks, padded_labels


def _load_pickle(f):
    with open(f, 'rb') as src:
        return pickle.load(src)


class SpectrogramDatasetMultiLabel(torch.utils.data.Dataset):
    """
    Multiple labels per example - more in the vein of FCNN labels.

    """

    def __init__(self,
                 files,
                 apply_log=True,
                 vertical_trim=0,
                 bootstrap_sample=False,
                 mask_beginning_and_end=False,
                 begin_mask=30,
                 end_mask=10):

        self.mask_beginning_and_end = mask_beginning_and_end
        self.apply_log = apply_log
        self.vertical_trim = vertical_trim
        self.bootstrap_sample = bootstrap_sample
        self.begin_mask = begin_mask
        self.end_mask = end_mask

        self.bootstrapped_files = np.random.choice(self.files, size=len(self.files),
                                                   replace=True) if self.bootstrap_sample else None
        self.files = self.bootstrapped_files if self.bootstrap_sample else files

    def __getitem__(self, idx):
        # returns a tuple with [0] the class label and [1] a slice of the spectrogram or the entire image.
        spect_slice, labels = _load_pickle(self.files[idx])
        spect_slice = spect_slice[self.vertical_trim:]
        spect_slice[spect_slice == 0] = 1
        if self.apply_log:
            spect_slice = np.log2(spect_slice)

        if self.mask_beginning_and_end and len(np.unique(labels)) == 1:
            labels[:self.begin_mask] = -1
            labels[-self.end_mask:] = -1

        return torch.tensor(spect_slice), torch.tensor(labels)

    def __len__(self):
        return len(self.files)

    def get_unique_labels(self):
        return self.unique_labels.keys()


class SpectrogramDatasetSingleLabel(torch.utils.data.Dataset):

    def __init__(self,
                 dataset_type,
                 data_path,
                 spect_type,
                 max_spec_length=40,
                 filtered_sounds=['C', 'Y'],
                 apply_log=True,
                 vertical_trim=0,
                 begin_cutoff_idx=0,
                 clip_spects=True,
                 bootstrap_sample=False):

        self.spect_lengths = defaultdict(list)
        self.dataset_type = dataset_type
        self.data_path = data_path
        self.spect_type = spect_type
        self.max_spec_length = max_spec_length
        self.filtered_sounds = filtered_sounds
        self.clip_spects = clip_spects
        self.apply_log = apply_log
        self.vertical_trim = vertical_trim
        self.begin_cutoff_idx = begin_cutoff_idx
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
                spect = spect[self.vertical_trim:, self.begin_cutoff_idx:]

                if self.apply_log:
                    # sometimes the mel spectrogram has all 0 filter banks.
                    # here we just set the 0s to 1s and then take the log
                    # transform, forcing those values back to 9. This removes
                    # the need for trimming rows to get rid of NaNs.
                    spect[spect == 0] = 1
                    spect = np.log2(spect)

                if spect.shape[1] >= self.max_spec_length:
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


if __name__ == '__main__':

    from glob import glob

    fpath = '/home/tc229954/data/beetles/extracted_data/train/mel_no_log_400_no_vert_trim/*'
    train_files = glob(fpath)
    s = SpectrogramDatasetMultiLabel(
        train_files,
        apply_log=True,
        vertical_trim=0,
        bootstrap_sample=False,
        mask_beginning_and_end=True)

    batch_size = 32
    d = torch.utils.data.DataLoader(s,
                                    batch_size=batch_size,
                                    num_workers=0,
                                    collate_fn=None if batch_size == 1 else pad_batch)
    for x, x_mask, y in d:
        print(x.shape, x_mask.shape, y.shape)
        break