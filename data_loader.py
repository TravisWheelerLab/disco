import time as t
import random
import math
import warnings
import torch
import numpy as np
import os

import spectrogram_analysis as sa


class Sound:
    def __init__(self,
                 sound_type,
                 file_name,
                 spectrogram_clip,
                 unix_time,
                 dataset="train_data"):
        self.sound_type = sound_type
        self.file_name = file_name
        self.spectrogram_clip = spectrogram_clip
        self.dataset = dataset
        self.trimmed = False
        self.unix_time = unix_time

    def save(self):
        # saves an individual sound as a .npy file in the path <data_set>/spect/ so they can be
        # read in by the neural network
        if isinstance(self.spectrogram_clip, torch.Tensor):
            self.spectrogram_clip = self.spectrogram_clip.numpy()
        np.save(
            os.path.join(self.dataset + "/spect", self.sound_type + "." + self.file_name + "." + str(self.unix_time)),
            self.spectrogram_clip)


def cutoff_kmeans_spectrograms(bf_obj, bci=45, eci=65):
    # locates the first spectrogram column of high intensity in a given spectrogram using 2-means clustering
    # and clips the spectrogram at that spot. This removes the milliseconds of human error background at the beginning
    # of a chirp that is not actually the chirp.
    sound_types = bf_obj.label_to_spectrogram.keys()
    bf_obj.fit_kmeans_subset(sound_types, 2, bci, eci)

    for sound_type, spectrogram_list in bf_obj.label_to_spectrogram.items():
        if sound_type != 'X' or 'Y' or 'C':
            for index, spect in enumerate(spectrogram_list):
                spect = spect.numpy()
                classified_points_list = bf_obj.classify_subset(spect, bci, eci)
                first_hi = None
                k = 0
                for point in classified_points_list:
                    if point:
                        first_hi = k
                        break
                    k += 1
                bf_obj.label_to_spectrogram[sound_type][index] = spect[:, first_hi:]


def save_all_sounds(sound_objects_list):
    # simple function that calls save for each sound file in a list of sound objects.
    for sound in sound_objects_list:
        sound.save()


def offload(beetle_object, file_wise, classes_excluded, train_pct, data_set):
    # saves each sound from each file into a sound object array depending on filewise/examplewise organization,
    # then saves entire array at the end.
    sound_objects = []
    if file_wise:
        for sound_type, spectrogram_list in beetle_object.label_to_spectrogram.items():
            for example in spectrogram_list:
                if isinstance(example, torch.Tensor):
                    example = example.numpy()
                if sound_type not in classes_excluded:
                    sound_objects.append(Sound(sound_type, beetle_object.name, example, t.time(), dataset=data_set))
        print(beetle_object.name, "done appending.")
    else:
        for sound_type, spectrogram_list in beetle_object.label_to_spectrogram.items():
            for example in spectrogram_list:
                if sound_type not in classes_excluded:
                    sound_objects.append(Sound(sound_type, beetle_object.name, example, t.time()))
        print(beetle_object.name, "done appending.")
        random.shuffle(sound_objects)
        test_data_cutoff = round(len(sound_objects) * (100 - train_pct) / 100)
        for example_index in range(test_data_cutoff):
            sound_objects[example_index].dataset = "test_data"
    save_all_sounds(sound_objects)


def determine_filewise_locations(filenames, train_pct):
    # if filewise, creates a boolean array that determines which file(s) will be test/train.
    test_set_marker = []
    n_test_files = max(1, math.floor((1 - train_pct / 100) * len(csvs_and_wav.keys())))
    for i in range(len(filenames)):
        if i < n_test_files:
            test_set_marker.append(True)
        else:
            test_set_marker.append(False)
    random.shuffle(test_set_marker)
    return test_set_marker


def offload_data(csvs_and_wav_files, file_wise=False, classes_excluded=[], cutoff=True, train_pct=80):
    # this function ensures that the function saving these files (offload) has the correct information about where each
    # sound from each file goes.
    test_set_marker = determine_filewise_locations(csvs_and_wav_files.keys(), train_pct)

    idx_test_arr = 0
    for filename, (wav, csv) in csvs_and_wav.items():
        spectrogram, label_to_spectrogram = sa.process_wav_file(wav, csv)
        file = sa.BeetleFile(filename, csv, wav, spectrogram, label_to_spectrogram)
        if cutoff and file != '1_M14F15_8_7':
            cutoff_kmeans_spectrograms(file)
        if file_wise:
            if test_set_marker[idx_test_arr]:
                data_set = "test_data"
            else:
                data_set = "train_data"
        else:
            data_set = None
        offload(file, file_wise, classes_excluded=classes_excluded, train_pct=train_pct, data_set=data_set)
        print(filename, "done processing.")
        idx_test_arr += 1


if __name__ == '__main__':
    begin_time = t.time()

    data_dir = './wav-files-and-annotations/'
    csvs_and_wav = sa.load_csv_and_wav_files_from_directory(data_dir)
    print("csvs and wav files loaded in.")

    offload_data(csvs_and_wav, file_wise=False, classes_excluded=['C', 'Y'])

    end_time = t.time()
    print('Elapsed time is %f seconds.' % (end_time - begin_time))
