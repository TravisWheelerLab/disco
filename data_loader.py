import matplotlib.colors
import time as t
import random
import math
import warnings

import spectrogram_analysis as sa


class Sound:
    def __init__(self,
                 sound_type,
                 file_name,
                 spectrogram_clip,
                 unix_time):
        self.sound_type = sound_type
        self.file_name = file_name
        self.spectrogram_clip = spectrogram_clip
        self.test_set = False
        self.trimmed = False
        self.unix_time = unix_time
        # self.trimmed_spectrogram = None

    def save(self):
        # string formatting - s before single quote insert variable {variable name} # dictionary
        pass


def split_train_test(beetle_files_dict, train_pct=80, file_wise=False):
    if file_wise:           # the case of reserving an extra file as the test data set.
        train, test = split_filewise(beetle_files_dict, train_pct)
    else:                   # the case of shuffling all examples within files, random chirps will go into each test set.
        train, test = split_examplewise(beetle_files_dict, train_pct)
    return train, test


def split_filewise(beetle_files_dict, train_pct):

    if len(beetle_files_dict) == 0:
        raise ValueError("File splitter expected at least 1 file but was given 0 files.")
    if len(beetle_files_dict) == 1:
        warnings.warn("File splitter expected more than 1 file but was given 1. Test and training data will not split.")

    test = []
    train = []
    test_files_num = max(1, math.floor((1-train_pct/100)*len(beetle_files_dict)))

    filename_keys = list(beetle_files_dict.keys())
    random.shuffle(filename_keys)

    for i in range(len(filename_keys)):
        file = filename_keys[i]
        beetlefile_object = beetle_files_dict[filename_keys[i]]
        for sound_type, spectrogram_list in beetlefile_object.label_to_spectrogram.items():
            for example in spectrogram_list:
                if i < test_files_num:
                    test.append(Sound(sound_type, file, example, t.time()))
                else:
                    train.append(Sound(sound_type, file, example, t.time()))
    return train, test


def split_examplewise(beetle_files_dict, train_pct):

    if len(beetle_files_dict) == 0:
        raise ValueError("File splitter expected at least 1 file but was given 0 files.")

    test = []
    sound_objects = []

    for file, beetlefile_object in beetle_files_dict.items():
        for sound_type, spectrogram_list in beetlefile_object.label_to_spectrogram.items():
            for example in spectrogram_list:
                sound_objects.append(Sound(sound_type, file, example, t.time()))
    random.shuffle(sound_objects)
    test_data_cutoff = round(len(sound_objects)*(100-train_pct)/100)
    for example_index in range(test_data_cutoff):
        test_set.append(sound_objects.pop(example_index))
        test_set[example_index].test_set = True
    train = sound_objects
    return train, test

if __name__ == '__main__':
    begin_time = t.time()

    data_dir = './wav-files-and-annotations/'
    csvs_and_wav = sa.load_csv_and_wav_files_from_directory(data_dir)

    beetle_files = {}

    for filename, (wav, csv) in csvs_and_wav.items():
        spectrogram, label_to_spectrogram = sa.process_wav_file(wav, csv)
        beetle_files[filename] = sa.BeetleFile(filename, csv, wav, spectrogram, label_to_spectrogram)

    train_set, test_set = split_train_test(beetle_files, file_wise=False)

    end_time = t.time()
    print('Elapsed time is %f seconds.' % (end_time - begin_time))