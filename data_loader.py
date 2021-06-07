

import numpy as np
import os
import pandas as pd

import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.model_selection import train_test_split

from collections import defaultdict
from glob import glob
from sklearn.cluster import KMeans

import spectrogram_analysis as sa


class Sound:
    def __init__(self,
                 sound_type,
                 file_name,
                 spectrogram_clip,
                 begin_wav_index = None,
                 end_wav_index = None):
        self.sound_type = sound_type
        self.file_name = file_name
        self.begin_wav_index = begin_wav_index
        self.end_wav_index = end_wav_index
        self.spectrogram_clip = spectrogram_clip
        self.test_set = False
        self.trimmed = False
        # self.trimmed_spectrogram = None

    #iterate over beetle files, then iterate over for key (filename), beetle file in dictionaryname.items() will iterae
    # items is a tuple of keys and values
        # sound types
            # actual sounds in those lists
    # string formatting - s before single quote insert variable {variable name} # dictionary


    def save(self):
        pass


def split_train_test(beetle_files, train_pct=80, file_wise=True):
    if file_wise:           # the case of reserving an extra file as the test data set.
        pass
    else:                   # the case of shuffling all examples within files, random chirps will go into each test set.
        x, x_test, y, y_test = train_test_split(x_train, labels, train_size=train_pct, random_state=42)

    # I need to return
    return

if __name__ == '__main__':
    # get beetle file object
    data_dir = './wav-files-and-annotations-1/'
    csvs_and_wav = load_csv_and_wav_files_from_directory(data_dir)

    beetle_files = {}

    for filename, (wav, csv) in csvs_and_wav.items():
        spectrogram, label_to_spectrogram = sa.process_wav_file(wav, csv)
        beetle_files[filename] = sa.BeetleFile(filename, csv, wav, spectrogram, label_to_spectrogram)


    train, test = split_train_test(beetle_files, train_pct=80)  # TODO
    # split train test can split on a file-wise basis or an example-wise basis
    # for examining different aspects of the data
    for beetle file in [test, train]:
        do_kmeans(beetle_file)  # have this
    remove_mislabeled_data(beetle_file)  # TODO
    # X = BACKGROUND
    For
    example in As, Bs, Xs, Ys, Cs:  # have this
    save(example)  # TODO