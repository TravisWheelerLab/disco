import os
from typing import Any

import pandas as pd
import torchaudio

from disco_sound.datasets import DataModule
from disco_sound.util.inference_utils import load_wav_file


class WhaleDataset(DataModule):
    def __init__(self, files, label_csv, n_fft, hop_length):

        self.files = list(files)
        self.label_mapping = pd.read_csv(label_csv, index_col=0)
        self.n_fft = n_fft
        self.hop_length = hop_length

    def collate_fn(self):
        return None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Any:
        # compute spectrogram
        example = self.files[index]
        wav, sample_rate = load_wav_file(example)
        bs = os.path.basename(example)
        label = self.label_mapping.loc[bs]["label"]
        spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )(wav).squeeze()
        return spectrogram, label
