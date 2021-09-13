import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from models import CNN1D


def plot_predictions_and_confidences(original_spectrogram,
                                     median_predictions,
                                     prediction_iqrs,
                                     save_prefix,
                                     n_samples=10,
                                     len_sample=300):
    len_spect = len_sample // 2
    inits = np.round(np.random.rand(n_samples) * median_predictions.shape[-1] - len_spect).astype(int)
    for i, center in enumerate(inits):
        fig, ax = plt.subplots(nrows=2, figsize=(13, 10), sharex=True)
        ax[0].imshow(original_spectrogram[:, center - len_spect:center + len_spect])
        med_slice = median_predictions[:, center - len_spect:center + len_spect]
        iqr_slice = prediction_iqrs[:, center - len_spect:center + len_spect]
        med_slice = np.transpose(med_slice)
        iqr_slice = np.transpose(iqr_slice)

        med_slice = np.expand_dims(med_slice, 0)
        iqr_slice = np.expand_dims(iqr_slice, 0)
        iqr_empty = np.zeros_like(iqr_slice)
        iqr_empty[:, :, 0] = np.sum(iqr_slice.squeeze(), axis=-1)
        ax[1].imshow(np.concatenate((med_slice, iqr_empty / np.max(iqr_empty)),
                                    axis=0), aspect='auto')
        ax[1].set_ylabel('iqrs              median predictions')
        ax[1].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_title('Predictions mapped to RGB values. red: A chirp, g: B chirp, b: background')
        ax[0].set_title('Random sample from spectrogram')
        ax[1].set_xlabel('spectrogram record')
        plt.savefig('{}_{}'.format(save_prefix, i))
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


def evaluate_spectrogram(spectrogram_dataset, models, tile_overlap, original_spectrogram_shape,
                         device='cpu'):

    assert_accuracy = False

    with torch.no_grad():
        medians_full_sequence = []
        iqrs_full_sequence = []
        if assert_accuracy:
            all_features = []

        for features in spectrogram_dataset:

            ensemble_preds = []
            features = features.to(device)

            for model in models:
                preds = torch.exp(model(features))
                ensemble_preds.append(preds.to('cpu').numpy())

            if assert_accuracy:
                all_features.extend(np.stack([seq[:, tile_overlap:-tile_overlap] for seq in features.to('cpu').numpy()]))

            ensemble_preds = np.stack([seq[:, :, tile_overlap:-tile_overlap] for seq in ensemble_preds])
            iqrs = np.zeros((ensemble_preds.shape[1], ensemble_preds.shape[2], ensemble_preds.shape[3]))
            medians = np.zeros((ensemble_preds.shape[1], ensemble_preds.shape[2], ensemble_preds.shape[3]))

            for class_idx in range(ensemble_preds.shape[2]):
                q75, q25 = np.percentile(ensemble_preds[:, :, class_idx, :], [75, 25], axis=0)
                median = np.median(ensemble_preds[:, :, class_idx, :], axis=0)
                iqrs[:, class_idx] = q75 - q25
                medians[:, class_idx] = median

            medians_full_sequence.extend(medians)
            iqrs_full_sequence.extend(iqrs)

    if assert_accuracy:
        all_features = np.concatenate(all_features, axis=-1)[:, :original_spectrogram_shape[-1]]
        print(all_features.shape, spectrogram_iterator.original_shape)
        print(np.all(all_features == spectrogram_iterator.original_spectrogram.numpy()))

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

        waveform, sample_rate = load_wav_file(self.wav_file)
        self.spectrogram = self.create_spectrogram(waveform, sample_rate)[vertical_trim:]
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
