import pickle

import numpy as np
import torch
import torchaudio

from disco_sound.datasets import DataModule
from disco_sound.util.inference_utils import load_wav_file


def pad_batch(batch, mask_flag=-1):
    """
    :param batch: The batch to pad.
    :param mask_flag: int. What character to interpret as the mask.
    :return: The padded batch.
    """
    features = [b[0] for b in batch]
    labels = [b[1] for b in batch]

    mxlen = np.max([f.shape[-1] for f in features])
    padded_batch = torch.zeros((len(batch), features[0].shape[0], mxlen))
    masks = torch.zeros((len(batch), 1, mxlen))
    padded_labels = torch.zeros((len(batch), mxlen), dtype=torch.int64) + mask_flag

    for i, (f, l) in enumerate(zip(features, labels)):
        padded_batch[i, :, : f.shape[-1]] = f
        masks[i, :, f.shape[-1] :] = True
        padded_labels[i, : l.shape[-1]] = l

    return padded_batch, masks.to(bool), padded_labels


def _load_pickle(f):
    """
    :param f: file containing the pickled object
    :return: the unpickled object
    """
    with open(f, "rb") as src:
        return pickle.load(src)


class SpectrogramDatasetMultiLabel(DataModule):
    """
    Handles potentially multiple labels per example.
    This class takes into account labels next to each other and therefore helps the neural network learn
    transitions between classes in training.
    """

    def collate_fn(self):
        return pad_batch

    def __init__(
        self,
        files,
        apply_log=True,
        vertical_trim=0,
        bootstrap_sample=False,
        mask_flag=-1,
        mask_beginning_and_end=False,
        begin_mask=None,
        end_mask=None,
    ):

        self.mask_beginning_and_end = mask_beginning_and_end
        if mask_beginning_and_end and (begin_mask is None or end_mask is None):
            raise ValueError(
                "If mask_beginning_and_end is true begin_mask and end_mask must"
                " not be None"
            )

        self.apply_log = apply_log
        self.mask_flag = mask_flag
        self.vertical_trim = vertical_trim
        self.bootstrap_sample = bootstrap_sample
        self.begin_mask = begin_mask
        self.end_mask = end_mask
        self.files = (
            np.random.choice(files, size=len(files), replace=True)
            if self.bootstrap_sample
            else files
        )
        # load all data into RAM before training
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
                    labels[self.begin_mask] = self.mask_flag
                    labels[-self.end_mask :] = self.mask_flag
                else:
                    # if it's not, throw it out. We don't want any possibility of bad data
                    # when training the model so we'll waste some compute.
                    labels[:] = self.mask_flag

        return torch.tensor(spect_slice), torch.tensor(labels)

    def __len__(self):
        """
        :return: The number of examples in the dataset.
        """
        return len(self.examples)

    def get_unique_labels(self):
        """
        :return: The number of unique labels in the dataset.
        """
        return self.unique_labels.keys()


class SpectrogramIterator(DataModule):
    def collate_fn(self):
        return None

    def __init__(
        self,
        tile_size,
        tile_overlap,
        vertical_trim,
        n_fft,
        hop_length,
        log_spect,
        mel_transform,
        wav_file=None,
        spectrogram=None,
    ):
        super().__init__()

        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        if self.tile_size <= tile_overlap:
            raise ValueError()

        self.wav_file = wav_file
        self.spectrogram = spectrogram

        if self.spectrogram is None and self.wav_file is None:
            raise ValueError("No spectrogram or .wav file specified.")

        self.vertical_trim = vertical_trim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.log_spect = log_spect
        self.mel_transform = mel_transform

        if self.spectrogram is None:
            waveform, self.sample_rate = load_wav_file(wav_file)
            self.spectrogram = self.create_spectrogram(waveform, self.sample_rate)

        self.spectrogram = self.spectrogram[vertical_trim:]

        if not torch.is_tensor(self.spectrogram):
            self.spectrogram = torch.tensor(self.spectrogram)

        if self.log_spect:
            self.spectrogram[self.spectrogram == 0] = 1
            self.spectrogram = self.spectrogram.log2()

        self.original_spectrogram = self.spectrogram.clone()
        self.original_shape = self.spectrogram.shape

        step_size = self.tile_size - 2 * self.tile_overlap
        leftover = self.spectrogram.shape[-1] % step_size
        # Since the length of our spectrogram % step_size isn't always 0, we will have a little
        # leftover at the end of spectrogram that we need to predict to get full coverage. There
        # are multiple ways to do this, but I decided to mirror pad the end of the spectrogram with
        # the correct amount of columns from the spectrogram so that padded_spectrogram % step_size == 0.
        # I cut off the predictions on the mirrored data after stitching the predictions together.
        to_pad = step_size - leftover + tile_size // 2

        if to_pad != 0:
            self.spectrogram = torch.cat(
                (
                    self.spectrogram,
                    torch.flip(self.spectrogram[:, -to_pad:], dims=[-1]),
                ),
                dim=-1,
            )

        self.indices = range(self.tile_size // 2, self.spectrogram.shape[-1], step_size)

        # mirror pad the beginning of the spectrogram
        self.spectrogram = torch.cat(
            (
                torch.flip(self.spectrogram[:, : self.tile_overlap], dims=[-1]),
                self.spectrogram,
            ),
            dim=-1,
        )

    def create_spectrogram(self, waveform, sample_rate):
        if self.mel_transform:
            spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
            )(waveform)
        else:
            spectrogram = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )(waveform)
        return spectrogram.squeeze()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        center_idx = self.indices[idx]
        # we want to overlap-tile starting from the beginning
        # so that our predictions are seamless.
        x = self.spectrogram[
            :, center_idx - self.tile_size // 2 : center_idx + self.tile_size // 2
        ]
        return x
