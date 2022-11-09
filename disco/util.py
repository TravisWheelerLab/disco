import logging

import torch

logger = logging.getLogger(__name__)

__all__ = ["add_white_noise", "add_gaussian_beeps"]


def add_white_noise(waveform, snr):
    rms_signal = torch.sqrt(torch.mean(waveform.squeeze() ** 2))
    std = torch.sqrt((rms_signal**2) / (10 ** (snr / 10)))
    noise = torch.randn_like(waveform) * std
    logger.info(f"Adding noise with SNR {snr} to waveform.")
    waveform += noise
    return waveform


def add_gaussian_beeps(waveform, sample_rate):
    x = torch.arange(waveform.shape[1])
    bep = torch.sin(2 * torch.pi * 440.0 * (x / sample_rate))
    # center a gaussian somewhere - here at 1000 units
    gaussian = torch.zeros_like(x).float()
    n_beeps = 10
    logger.info(f"Adding {n_beeps} beeps to waveform.")

    for _ in range(n_beeps):
        width = 1000 + int(np.random.rand() * 10000)
        center = int(np.random.rand() * waveform.shape[1]) - 10000
        gaussian += np.exp(-((x - center) ** 2) / (width**2))
    waveform += gaussian * bep

    return waveform