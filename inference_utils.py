import torch
import torchaudio


def load_wav_file(wav_filename):
    waveform, sample_rate = torchaudio.load(wav_filename)
    return waveform, sample_rate


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
        self.spectrogram = self.create_spectrogram(waveform, sample_rate)
        self.original_shape = self.spectrogram.shape

        # reverse pad the beginning of the spectrogram
        # TODO: stop being lazy and figure out how much to clip off then
        # end to get the same size arrays
        self.spectrogram = torch.cat((self.spectrogram,
                                      torch.flip(self.spectrogram[:, -self.tile_size:], dims=[-1])),
                                     dim=-1)

        self.indices = range(self.tile_size // 2, self.spectrogram.shape[-1],
                             self.tile_size - 2 * self.tile_overlap)
        # reverse-pad the beginning of the spectrogram
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
