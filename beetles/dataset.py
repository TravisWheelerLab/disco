import os
import random
import pickle
from collections import defaultdict, OrderedDict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch

from beetles import INDEX_TO_LABEL, LABEL_TO_INDEX, MASK_FLAG


def pad_batch(batch):
    features = [b[0] for b in batch]
    labels = [b[1] for b in batch]

    mxlen = np.max([f.shape[-1] for f in features])
    padded_batch = torch.zeros((len(batch), features[0].shape[0], mxlen))
    masks = torch.zeros((len(batch), 1, mxlen))
    padded_labels = torch.zeros((len(batch), mxlen), dtype=torch.int64) + MASK_FLAG

    for i, (f, l) in enumerate(zip(features, labels)):
        padded_batch[i, :, : f.shape[-1]] = f
        masks[i, :, f.shape[-1] :] = True
        padded_labels[i, : l.shape[-1]] = l

    return padded_batch, masks.to(bool), padded_labels


def _load_pickle(f):
    with open(f, "rb") as src:
        return pickle.load(src)


class SpectrogramDatasetMultiLabel(torch.utils.data.Dataset):
    """
    Multiple labels per example - more similar to FCNN labels.
    """

    def __init__(
        self,
        files,
        apply_log=True,
        vertical_trim=0,
        bootstrap_sample=False,
        mask_beginning_and_end=False,
        begin_mask=None,
        end_mask=None,
    ):

        self.mask_beginning_and_end = mask_beginning_and_end
        if mask_beginning_and_end and (begin_mask is None or end_mask is None):
            raise ValueError(
                "If mask_beginning_and_end is true begin_mask and end_mask must"
                "not be None"
            )

        self.apply_log = apply_log
        self.vertical_trim = vertical_trim
        self.bootstrap_sample = bootstrap_sample
        self.begin_mask = begin_mask
        self.end_mask = end_mask
        self.files = files
        self.files = (
            np.random.choice(self.files, size=len(self.files), replace=True)
            if self.bootstrap_sample
            else self.files
        )
        self.examples = [_load_pickle(f) for f in self.files]

    def __getitem__(self, idx):

        spect_slice, labels = self.examples[idx]
        spect_slice = spect_slice[self.vertical_trim :]

        if self.apply_log:
            # take care of NaNs after taking the log.
            spect_slice[spect_slice == 0] = 1
            spect_slice = np.log2(spect_slice)

        if self.mask_beginning_and_end:
            if len(np.unique(labels)) == 1:
                # if there's only one class
                if labels.shape[0] > (self.begin_mask + self.end_mask):
                    # and if the label vector is longer than where we're supposed to mask
                    labels[self.begin_mask] = MASK_FLAG
                    labels[-self.end_mask :] = MASK_FLAG
                else:
                    # if it's not, throw it out. We don't want any possibility of bad data
                    # when training the model so we'll waste some compute.
                    labels[:] = MASK_FLAG

        return torch.tensor(spect_slice), torch.tensor(labels)

    def __len__(self):
        return len(self.examples)

    def get_unique_labels(self):
        return self.unique_labels.keys()


class SpectrogramDatasetSingleLabel(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_type,
        data_path,
        spect_type,
        max_spec_length=40,
        filtered_sounds=["C", "Y"],
        apply_log=True,
        vertical_trim=0,
        begin_cutoff_idx=0,
        clip_spects=True,
        bootstrap_sample=False,
    ):

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
        self.spectrograms_list, self.unique_labels = self.load_in_all_files(
            self.dataset_type,
            self.spect_type,
            self.filtered_sounds,
            self.bootstrap_sample,
        )

    def load_in_all_files(
        self, dataset_type, spect_type, filtered_labels, bootstrap_sample
    ):

        spectrograms_list = []
        root = os.path.join(self.data_path, dataset_type, spect_type, "spect")
        files = glob(os.path.join(root, "*"))
        class_counter = defaultdict(int)
        for filepath in files:
            head, tail = os.path.split(filepath)
            label = tail.split(".")[0]
            spect = np.load(filepath)
            if label not in filtered_labels and spect.shape[1] >= self.max_spec_length:
                class_counter[label] += 1
                spect = spect[self.vertical_trim :, self.begin_cutoff_idx :]

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
            indices = np.random.choice(
                len(spectrograms_list), size=len(spectrograms_list), replace=True
            )
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
            spect_slice = torch.tensor(
                spect[:, random_index : random_index + spect_size]
            )
            label_tensor = torch.tensor(
                np.repeat(a=LABEL_TO_INDEX[label], repeats=spect_size)
            )
        else:
            spect_slice = torch.tensor(spect)
            label_tensor = torch.tensor(
                np.repeat(a=LABEL_TO_INDEX[label], repeats=len(spect[1]))
            )
        return spect_slice, label_tensor

    def __len__(self):
        return len(self.spectrograms_list)

    def get_unique_labels(self):
        return self.unique_labels.keys()
