import os
import pdb
import torch
import torchaudio
import numpy as np
import pandas as pd
import pickle
np.random.seed(0)
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict

from models import CNN1D
from hmm_process import load_in_hmm

CLASS_CODE_TO_NAME = {0: 'A', 1: 'B', 2: 'BACKGROUND'}
NAME_TO_CLASS_CODE = {v: k for k, v in CLASS_CODE_TO_NAME.items()}
SOUND_TYPE_TO_COLOR = {'A': 'r', 'B': 'g', 'BACKGROUND': 'k'}


def aggregate_predictions(predictions):
    if predictions.ndim != 1:
        raise ValueError('expected array of size N, got {}'.format(predictions.shape))

    diff = np.diff(predictions)  # transition regions will be nonzero
    idx, = diff.nonzero()
    current_class = predictions[0]
    current_idx = 0
    class_idx_to_prediction_start_and_end = defaultdict(list)

    for i in range(len(idx)):
        class_idx_to_prediction_start_and_end[current_class].append([current_idx, idx[i]])
        current_class = predictions[idx[i]+1]
        current_idx = idx[i]+1

    return class_idx_to_prediction_start_and_end



def convert_spectrogram_index_to_seconds(spect_idx, hop_length, sample_rate):
    seconds_per_hop = hop_length / sample_rate
    return spect_idx*seconds_per_hop


def pickle_data(data, path):
    with open(path, 'wb') as dst:
        pickle.dump(data, dst)


def load_pickle(path):
    with open(path, 'rb') as src:
        data = pickle.load(src)
    return data


def save_csv_from_predictions(output_csv_path, predictions, sample_rate, hop_length):
    # We just need to get the beginning and end of each chirp and convert those
    # to seconds.
    class_idx_to_prediction_start_end = aggregate_predictions(predictions)
    # window size by default is n_fft. hop_length is interval b/t consecutive spectrograms
    # i don't think padding is performed when the spectrogram is calculated
    list_of_dicts_for_dataframe = []
    for class_idx, starts_and_ends in class_idx_to_prediction_start_end.items():
        for start, end in starts_and_ends:
            dataframe_dict = {'Sound_Type': CLASS_CODE_TO_NAME[class_idx],
                              'Begin Time (s)': convert_spectrogram_index_to_seconds(start,
                                                                                     hop_length=hop_length,
                                                                                     sample_rate=sample_rate),
                              'End Time (s)': convert_spectrogram_index_to_seconds(end,
                                                                                   hop_length=hop_length,
                                                                                   sample_rate=sample_rate)}
            list_of_dicts_for_dataframe.append(dataframe_dict)
    df = pd.DataFrame.from_dict(list_of_dicts_for_dataframe)
    df.to_csv(output_csv_path)
    return df


def run_hmm(predictions):
    """
    :param predictions: np array of point-wise argmaxed predictions (size N).
    :return: smoothed predictions
    """
    if predictions.ndim != 1:
        raise ValueError('expected array of size N, got {}'.format(predictions.shape))
    hmm = load_in_hmm()

    # forget about the first element b/c it's the start state
    smoothed_predictions = np.asarray(hmm.predict(sequence=predictions, algorithm='viterbi')[1:])
    return smoothed_predictions


def convert_argmaxed_array_to_rgb(predictions):
    # data: array of shape 1xN
    rgb = np.zeros((1, predictions.shape[-1], 3))
    for class_idx in CLASS_CODE_TO_NAME.keys():
        rgb[:, np.where(predictions == class_idx), class_idx] = 1
    return rgb


def convert_time_to_spect_index(time, hop_length, sample_rate):
    return np.round(time * sample_rate).astype(np.int) // hop_length


def load_prediction_csv(csv_path, hop_length, sample_rate):
    df = pd.read_csv(csv_path)
    df['Begin Spect Index'] = [convert_time_to_spect_index(x, hop_length, sample_rate) for x in df['Begin Time (s)']]
    df['End Spect Index'] = [convert_time_to_spect_index(x, hop_length, sample_rate) for x in df['End Time (s)']]
    return df


def plot_predictions_and_confidences(original_spectrogram,
                                     median_predictions,
                                     prediction_iqrs,
                                     hmm_predictions,
                                     processed_predictions,
                                     save_prefix,
                                     n_samples=10,
                                     len_sample=1000):
    len_spect = len_sample // 2
    inits = np.round(np.random.rand(n_samples) * median_predictions.shape[-1] - len_spect).astype(int)
    # oof lots of plotting code
    for i, center in enumerate(inits):
        fig, ax = plt.subplots(nrows=2, figsize=(13, 10), sharex=True)
        ax[0].imshow(original_spectrogram[:, center - len_spect:center + len_spect])
        med_slice = median_predictions[:, center - len_spect:center + len_spect]
        iqr_slice = prediction_iqrs[:, center - len_spect:center + len_spect]
        hmm_slice = hmm_predictions[center - len_spect:center + len_spect]
        processed_slice = processed_predictions[center - len_spect:center + len_spect]
        med_slice = np.transpose(med_slice)
        iqr_slice = np.transpose(iqr_slice)

        med_slice = np.expand_dims(med_slice, 0)
        iqr_slice = np.expand_dims(iqr_slice, 0)

        hmm_rgb = convert_argmaxed_array_to_rgb(hmm_slice)
        processed_rgb = convert_argmaxed_array_to_rgb(processed_slice)

        ax[1].imshow(np.concatenate((med_slice,
                                     iqr_slice,  # will there be problems with normalization here?
                                     processed_rgb,
                                     hmm_rgb),
                                    axis=0),
                     aspect='auto', interpolation='nearest')

        ax[1].set_yticks([0, 1, 2, 3])
        ax[1].set_yticklabels(['median prediction', 'iqr', 'thresholded predictions',
                               'smoothed w/ hmm'])

        ax[1].set_xticks([])
        ax[1].set_title('Predictions mapped to RGB values. red: A chirp, green: B chirp, blue: background')
        ax[0].set_title('Random sample from spectrogram')
        ax[1].set_xlabel('spectrogram record')
        print('{}_{}.png'.format(save_prefix, i))
        plt.savefig('{}_{}.png'.format(save_prefix, i))
        plt.close()


def assemble_ensemble(model_directory, model_extension, device,
                      in_channels):
    model_paths = glob(os.path.join(model_directory, "*" + model_extension))
    if not len(model_paths):
        raise ValueError("no models found at {}".format(os.path.join(model_directory,
                                                                     "*" + model_extension)))
    models = []
    for model_path in model_paths:
        skeleton = CNN1D(in_channels).to(device)
        skeleton.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        models.append(skeleton.eval())

    return models


def load_wav_file(wav_filename):
    waveform, sample_rate = torchaudio.load(wav_filename)
    return waveform, sample_rate


def predict_with_ensemble(ensemble, features):
    ensemble_preds = []

    for model in ensemble:
        preds = torch.exp(model(features))
        ensemble_preds.append(preds.to('cpu').numpy())

    return ensemble_preds


def calculate_median_and_iqr(ensemble_preds):
    # TODO: add docstring

    iqrs = np.zeros((ensemble_preds.shape[1], ensemble_preds.shape[2], ensemble_preds.shape[3]))
    medians = np.zeros((ensemble_preds.shape[1], ensemble_preds.shape[2], ensemble_preds.shape[3]))
    for class_idx in range(ensemble_preds.shape[2]):
        q75, q25 = np.percentile(ensemble_preds[:, :, class_idx, :], [75, 25], axis=0)
        median = np.median(ensemble_preds[:, :, class_idx, :], axis=0)
        iqrs[:, class_idx] = q75 - q25
        medians[:, class_idx] = median

    return iqrs, medians


def evaluate_spectrogram(spectrogram_dataset,
                         models,
                         tile_overlap,
                         original_spectrogram_shape,
                         device='cpu'):
    assert_accuracy = False

    with torch.no_grad():
        medians_full_sequence = []
        iqrs_full_sequence = []
        if assert_accuracy:
            all_features = []

        for features in spectrogram_dataset:

            features = features.to(device)
            ensemble_preds = predict_with_ensemble(models, features)

            if assert_accuracy:
                all_features.extend(
                    np.stack([seq[:, tile_overlap:-tile_overlap] for seq in features.to('cpu').numpy()]))

            ensemble_preds = np.stack([seq[:, :, tile_overlap:-tile_overlap] for seq in ensemble_preds])
            iqrs, medians = calculate_median_and_iqr(ensemble_preds)
            medians_full_sequence.extend(medians)
            iqrs_full_sequence.extend(iqrs)

    if assert_accuracy:
        all_features = np.concatenate(all_features, axis=-1)[:, :original_spectrogram_shape[-1]]
        assert(np.all(all_features == spectrogram_iterator.original_spectrogram.numpy()))

    medians_full_sequence = np.concatenate(medians_full_sequence, axis=-1)[:, :original_spectrogram_shape[-1]]
    iqrs_full_sequence = np.concatenate(iqrs_full_sequence, axis=-1)[:, :original_spectrogram_shape[-1]]

    return medians_full_sequence, iqrs_full_sequence


class SpectrogramIterator(torch.nn.Module):
    # TODO: replace args in __init__ with sa.form_spectrogram_type
    def __init__(self,
                 tile_size,
                 tile_overlap,
                 wav_file,
                 vertical_trim,
                 n_fft,
                 hop_length,
                 log_spect,
                 mel_transform
                 ):

        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        if self.tile_size <= tile_overlap:
            raise ValueError()
        self.wav_file = wav_file
        self.vertical_trim = vertical_trim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.log_spect = log_spect
        self.mel_transform = mel_transform

        waveform, self.sample_rate = load_wav_file(self.wav_file)
        self.spectrogram = self.create_spectrogram(waveform, self.sample_rate)[vertical_trim:]
        self.original_spectrogram = self.spectrogram.clone()
        self.original_shape = self.spectrogram.shape

        step_size = self.tile_size - 2 * self.tile_overlap
        leftover = self.spectrogram.shape[-1] % step_size
        # Since the length of our spectrogram % step_size isn't always 0, we will have a little
        # leftover at the end of spectrogram that we need to predict to get full coverage. There
        # are multiple ways to do this but I decided to mirror pad the end of the spectrogram with
        # the correct amount of columns from the spectrogram so that padded_spectrogram % step_size == 0.
        # I cut off the predictions on the mirrored data after stitching the predictions together.
        to_pad = step_size - leftover + tile_size // 2

        if to_pad != 0:
            self.spectrogram = torch.cat((self.spectrogram,
                                          torch.flip(self.spectrogram[:, -to_pad:], dims=[-1])),
                                         dim=-1)

        self.indices = range(self.tile_size // 2, self.spectrogram.shape[-1],
                             step_size)

        # mirror pad the beginning of the spectrogram
        self.spectrogram = torch.cat((torch.flip(self.spectrogram[:, :self.tile_overlap], dims=[-1]),
                                      self.spectrogram), dim=-1)

    def create_spectrogram(self, waveform, sample_rate):
        if self.mel_transform:
            spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                               n_fft=self.n_fft,
                                                               hop_length=self.hop_length)(waveform)
        else:
            spectrogram = torchaudio.transforms.Spectrogram(sample_rate=self.sample_rate,
                                                            n_fft=self.n_fft,
                                                            hop_length=self.hop_length)(waveform)
        if self.log_spect:
            spectrogram = spectrogram.log2()

        return spectrogram.squeeze()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        center_idx = self.indices[idx]
        # we want to overlap-tile starting from the beginning
        # so that our predictions are seamless.
        x = self.spectrogram[:, center_idx - self.tile_size // 2: center_idx + self.tile_size // 2]
        # print(center_idx, x.shape, center_idx-self.tile_size//2, center_idx+self.tile_size//2)
        return x
