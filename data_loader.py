import time as t
import random
import math
import warnings
import torch

import spectrogram_analysis as sa


class Sound:
    def __init__(self,
                 sound_type,
                 file_name,
                 spectrogram_clip,
                 unix_time,
                 dataset="train"):
        self.sound_type = sound_type
        self.file_name = file_name
        self.spectrogram_clip = spectrogram_clip
        self.dataset = dataset
        self.trimmed = False
        self.unix_time = unix_time

    def save(self):
        if isinstance(self.spectrogram_clip, torch.Tensor):
            self.spectrogram_clip = self.spectrogram_clip.numpy()
        # todo: look up string formatting
        pass


def split_train_test(beetle_files_dict, train_pct=80, file_wise=False, k_means_cutoff=False, bci=45, eci=65):
    # places beetle files' individual labeled sounds into an array holding marked train and test labels
    # designated by each sound object's "dataset" attribute so they can eventually be fed into directories
    if file_wise:  # the case of reserving an extra file as the test data set
        sound_objects = split_filewise(beetle_files_dict, train_pct, cutoff=k_means_cutoff, bci=bci, eci=eci)
    else:  # the case of shuffling all examples within files, random chirps will go into each test set
        sound_objects = split_examplewise(beetle_files_dict, train_pct, cutoff=k_means_cutoff, bci=bci, eci=eci)
    return sound_objects


def split_filewise(beetle_files_dict, train_pct, cutoff, bci, eci):
    # used if it is more desirable to reserve an entire beetle courtship as the test data rather than examples from
    # multiple courtships that the neural network would have seen before
    if len(beetle_files_dict) == 0:
        raise ValueError("File splitter expected at least 1 file but was given 0 files.")
    if len(beetle_files_dict) == 1:
        warnings.warn("File splitter expected more than 1 file but was given 1. Test and training data will not split.")

    sound_objects = []
    test_files_num = max(1, math.floor((1 - train_pct / 100) * len(beetle_files_dict)))

    filename_keys = list(beetle_files_dict.keys())
    random.shuffle(filename_keys)

    for i in range(len(filename_keys)):
        file = filename_keys[i]
        beetlefile_object = beetle_files_dict[filename_keys[i]]
        if cutoff and file != '1_M14F15_8_7':
            cutoff_kmeans_spectrograms(beetlefile_object, bci, eci)
        for sound_type, spectrogram_list in beetlefile_object.label_to_spectrogram.items():
            for example in spectrogram_list:
                if isinstance(example, torch.Tensor):
                    example = example.numpy()
                if i < test_files_num:
                    sound_objects.append(Sound(sound_type, file, example, t.time(), dataset="test"))
                else:
                    sound_objects.append(Sound(sound_type, file, example, t.time()))
    return sound_objects


def split_examplewise(beetle_files_dict, train_pct, cutoff, bci, eci):
    # used if it is more desirable to shuffle all chirps across beetles and split them into the train and test sets
    if len(beetle_files_dict) == 0:
        raise ValueError("File splitter expected at least 1 file but was given 0 files.")

    sound_objects = []

    for file, beetlefile_object in beetle_files_dict.items():
        if cutoff and file != '1_M14F15_8_7':
            cutoff_kmeans_spectrograms(beetlefile_object, bci, eci)
        for sound_type, spectrogram_list in beetlefile_object.label_to_spectrogram.items():
            for example in spectrogram_list:
                sound_objects.append(Sound(sound_type, file, example, t.time()))
    random.shuffle(sound_objects)
    test_data_cutoff = round(len(sound_objects) * (100 - train_pct) / 100)
    for example_index in range(test_data_cutoff):
        sound_objects[example_index].dataset = "test"

    return sound_objects


def cutoff_kmeans_spectrograms(bf_obj, bci, eci):
    # locates the first spectrogram column of high intensity in a given spectrogram using 2-means clustering
    # and clips the spectrogram at that spot. This removes the milliseconds of human error background at the beginning
    # of a chirp that is not actually the chirp.
    sound_types = bf_obj.label_to_spectrogram.keys()
    bf_obj.fit_kmeans_subset(sound_types, 2, bci, eci)

    for sound_type, spectrogram_list in bf_obj.label_to_spectrogram.items():
        if sound_type != 'X' or 'Y':

            # length_of_list = len(spectrogram_list[0])
            # for i in range(len(spectrogram_list[0])):
            #     spectrogram_list[i] = spectrogram_list[i].numpy()
            #     classified_points_list = bf_obj.classify_subset(spectrogram_list[i], bci, eci)
            #     first_hi = None
            #     k = 0
            #     for point in classified_points_list:
            #         if point:
            #             first_hi = k
            #             break
            #         k += 1
            #     spectrogram_list[i] = spectrogram_list[i][:, first_hi:]

            i = 0
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


if __name__ == '__main__':
    begin_time = t.time()

    data_dir = './wav-files-and-annotations-1/'
    csvs_and_wav = sa.load_csv_and_wav_files_from_directory(data_dir)

    beetle_files = {}

    for filename, (wav, csv) in csvs_and_wav.items():
        spectrogram, label_to_spectrogram = sa.process_wav_file(wav, csv)
        beetle_files[filename] = sa.BeetleFile(filename, csv, wav, spectrogram, label_to_spectrogram)

    sounds = split_train_test(beetle_files, file_wise=False, k_means_cutoff=True)

    end_time = t.time()
    print('Elapsed time is %f seconds.' % (end_time - begin_time))
