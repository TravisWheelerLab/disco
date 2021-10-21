import time
import random
import math
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pickle
import os

import spectrogram_analysis as sa
from data_feeder import INDEX_TO_LABEL, LABEL_TO_INDEX


class Sound:

    def __init__(self,
                 data_out_path,
                 sound_type,
                 file_name,
                 spectrogram_clip,
                 unix_time,
                 spectrogram_type,
                 dataset="train"):

        self.data_out_path = data_out_path
        self.sound_type = sound_type
        self.file_name = file_name
        self.spectrogram_clip = spectrogram_clip
        self.dataset = dataset
        self.trimmed = False
        self.unix_time = unix_time
        self.spectrogram_type = spectrogram_type

    def save(self):
        # saves an individual sound as a .npy file in the path <data_set>/spect/ so they can be
        # read in by the neural network
        if isinstance(self.spectrogram_clip, torch.Tensor):
            self.spectrogram_clip = self.spectrogram_clip.numpy()
        filename = self.sound_type + '.' + self.file_name + '.' + str(self.unix_time)
        save_dir = os.path.join(self.data_out_path, self.dataset, self.spectrogram_type)
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        np.save(filepath, self.spectrogram_clip)


def offload_data(data_out_path, csvs_and_wav_files, cutoff, mel, log, n_fft, vert_trim, file_wise=False,
                 classes_excluded=[], train_pct=80):
    # this function ensures that the function saving these files (offload) has the correct information about where each
    # sound from each file goes.
    test_set_marker = determine_filewise_locations(csvs_and_wav_files.keys(), train_pct)

    idx_test_arr = 0
    for filename, (wav, csv) in csvs_and_wav.items():
        spectrogram, label_to_spectrogram = sa.process_wav_file(wav, csv, n_fft, mel, log, vert_trim)
        file = sa.BeetleFile(filename, csv, wav, spectrogram, label_to_spectrogram, mel, n_fft, log, vert_trim)
        if cutoff and file != '1_M14F15_8_7':
            cutoff_kmeans_spectrograms(file)
        if file_wise:
            if test_set_marker[idx_test_arr]:
                data_set = "test"
            else:
                data_set = "train"
        else:
            data_set = None

        offload(data_out_path, file, file_wise, classes_excluded=classes_excluded, train_pct=train_pct,
                data_set=data_set,
                spectrogram_type=file.spectrogram_type)
        print(filename, "done processing.")
        idx_test_arr += 1


# need to save contiguous labels.
# But first I want to visualize all of the data I labeled.
def determine_filewise_locations(filenames, train_pct):
    # if filewise saving scheme, creates a boolean array that determines which file(s) will be test/train.
    test_set_marker = []
    n_test_files = max(1, math.floor((1 - train_pct / 100) * len(csvs_and_wav.keys())))
    for i in range(len(filenames)):
        if i < n_test_files:
            test_set_marker.append(True)
        else:
            test_set_marker.append(False)
    random.shuffle(test_set_marker)
    return test_set_marker


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


def offload(data_out_path,
            beetle_object,
            file_wise,
            classes_excluded,
            train_pct,
            data_set,
            spectrogram_type):
    # saves each sound from each file into a sound object array depending on filewise/examplewise organization,
    # then saves entire array at the end.
    sound_objects = []
    if file_wise:
        for sound_type, spectrogram_list in beetle_object.label_to_spectrogram.items():
            for example in spectrogram_list:
                if isinstance(example, torch.Tensor):
                    example = example.numpy()
                if sound_type not in classes_excluded:
                    sound_objects.append(
                        Sound(data_out_path, sound_type, beetle_object.name, example, time.time(), dataset=data_set,
                              spectrogram_type=spectrogram_type))
        print(beetle_object.name, "done appending.")
    else:
        for sound_type, spectrogram_list in beetle_object.label_to_spectrogram.items():
            for example in spectrogram_list:
                if sound_type not in classes_excluded:
                    sound_objects.append(Sound(data_out_path, sound_type, beetle_object.name, example, time.time(),
                                               spectrogram_type=spectrogram_type))
        print(beetle_object.name, "done appending.")
        random.shuffle(sound_objects)
        test_data_cutoff = round(len(sound_objects) * (100 - train_pct) / 100)
        for example_index in range(test_data_cutoff):
            sound_objects[example_index].dataset = "test"
            sound_objects[example_index].spectrogram_type = spectrogram_type
    save_all_sounds(sound_objects)


def save_all_sounds(sound_objects_list):
    # simple function that calls save for each sound file in a list of sound objects.
    for sound in sound_objects_list:
        sound.save()
    print("sounds saved.")


def create_directories(spectrogram_type):
    datasets = ['train', 'test', 'validation']
    for dataset in datasets:
        directory_path = os.path.join('data', dataset, spectrogram_type, 'spect')
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)


def save_data(out_path, data_list):

    os.makedirs(out_path, exist_ok=True)

    for i, (features, label_vector) in enumerate(data_list):

        if label_vector.shape[0] > 10000:
            # way too big
            continue
        uniq = np.unique(label_vector, return_counts=True)
        label = np.argmax(uniq[1])
        if INDEX_TO_LABEL[label] == 'X' and len(uniq[0]) != 1:
            lvec = label_vector[label_vector != LABEL_TO_INDEX['BACKGROUND']]
            uniq = np.unique(lvec, return_counts=True)
            label = np.argmax(uniq[1])

        out_fpath = os.path.join(out_path,
                                 INDEX_TO_LABEL[label] + '_' + str(i) + '.pkl')

        with open(out_fpath, 'wb') as dst:
            pickle.dump([features.numpy(), label_vector], dst)


if __name__ == '__main__':
    import sys
    # have to set seed for reproducibility.
    random.seed(0)
    np.random.seed(0)

    mel = True
    log = False
    n_fft = int(sys.argv[1])
    vert_trim = 0
    cutoff = False
    file_wise = False

    spect_type = sa.form_spectrogram_type(mel, n_fft, log, vert_trim)
    print(spect_type)

    data_dir = '/home/tc229954/data/beetles/more_labeled_training_data'

    csv_and_wav = sa.load_csv_and_wav_files_from_directory(data_dir)

    out = []

    for filename, (wav, csv) in csv_and_wav.items():
        features_and_labels = sa.process_wav_file(wav, csv, n_fft, mel, log, vert_trim)
        out.extend(features_and_labels)

    random.shuffle(out)
    indices = np.arange(len(out))
    train_idx, test_idx, _, _ = train_test_split(indices, indices, test_size=0.15,
                                                 random_state=0)
    test_idx, val_idx, _, _ = train_test_split(test_idx, test_idx, test_size=0.5,
                                               random_state=0)

    train_split = np.asarray(out)[train_idx]
    val_split = np.asarray(out)[val_idx]
    test_split = np.asarray(out)[test_idx]

    train_path = '/home/tc229954/data/beetles/extracted_data/train/{}'.format(spect_type)
    validation_path = '/home/tc229954/data/beetles/extracted_data/validation/{}'.format(spect_type)
    test_path = '/home/tc229954/data/beetles/extracted_data/test/{}'.format(spect_type)

    save_data(train_path, train_split)
    save_data(validation_path, val_split)
    save_data(test_path, test_split)
