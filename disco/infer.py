import pdb
import warnings
import numpy as np
import os
import logging
import torch

from argparse import ArgumentParser

import disco.heuristics as heuristics
import disco.inference_utils as infer
from disco.config import Config

# get rid of torchaudio warning us that our spectrogram calculation needs different parameters
warnings.filterwarnings("ignore", category=UserWarning)

log = logging.getLogger(__name__)


def run_inference(
    config,
    wav_file=None,
    output_csv_path=None,
    saved_model_directory=None,
    model_extension=".pt",
    tile_overlap=128,
    tile_size=1024,
    batch_size=32,
    input_channels=108,
    hop_length=200,
    vertical_trim=20,
    n_fft=1150,
    debug=None,
    num_threads=4,
):
    """
    Script to run the inference routine. Briefly: Model ensemble is loaded in, used to evaluate the spectrogram, and
    heuristics and the hmm are applied to the ensemble's predictions. The .csv containing model labels is saved and
    debugging information is saved depending on whether or not debug is a string or None.

    The ensemble predicts a .wav file quickly and seamlessly by using an overlap-tile strategy.

    :param config: disco.Config() object.
    :param wav_file: str. .wav file to analyze.
    :param output_csv_path: str. Where to save predictions.
    :param saved_model_directory: str. Where models are saved.
    :param model_extension: str. Model file suffix. Default ".pt".
    :param tile_overlap: How much to overlap subsequent evaluation windows.
    :param tile_size: Size of tiles ingested into ensemble.
    :param batch_size: How many tiles to evaluate in parallel.
    :param input_channels: Number of input channels for the model ensemble.
    :param hop_length: Used in spectrogram calculation.
    :param vertical_trim: How many rows to chop off from the beginning of the spectrogram (in effect, a high-pass filter).
    :param n_fft: N ffts to use when calulating the spectrogram.
    :param debug: str. Whether or not to save debugging data. None: Don't save, str: save in "str".
    :param num_threads: How many threads to use when loading data.
    :return: None. Everything relevant is saved to disk.
    """

    if tile_size % 2 != 0:
        raise ValueError("tile_size must be even, got {}".format(tile_size))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(num_threads)
    models = infer.assemble_ensemble(
        saved_model_directory, model_extension, device, input_channels, config
    )
    if len(models) < 1:
        raise ValueError(
            "expected more than 1 model, found {}. Is the model directory and extension correct?".format(
                len(models)
            )
        )

    spectrogram_iterator = infer.SpectrogramIterator(
        tile_size,
        tile_overlap,
        vertical_trim=vertical_trim,
        n_fft=n_fft,
        hop_length=hop_length,
        log_spect=True,
        mel_transform=True,
        wav_file=wav_file,
    )
    spectrogram_dataset = torch.utils.data.DataLoader(
        spectrogram_iterator, shuffle=False, batch_size=batch_size, drop_last=False
    )

    medians, iqr = infer.evaluate_spectrogram(
        spectrogram_dataset,
        models,
        tile_overlap,
        spectrogram_iterator.original_spectrogram,
        spectrogram_iterator.original_shape,
        device=device,
    )

    predictions = np.argmax(medians, axis=0).squeeze()

    for heuristic in heuristics.HEURISTIC_FNS:
        log.info(f"applying heuristic function {heuristic.__name__}")
        predictions = heuristic(predictions, iqr, config.name_to_class_code)

    hmm_predictions = infer.smooth_predictions_with_hmm(predictions, config)

    if output_csv_path is not None:
        _, ext = os.path.splitext(output_csv_path)

        if not ext:
            output_csv_path = f"{output_csv_path}.csv"

        infer.save_csv_from_predictions(
            output_csv_path,
            hmm_predictions,
            sample_rate=spectrogram_iterator.sample_rate,
            hop_length=hop_length,
            name_to_class_code=config.name_to_class_code,
        )

    if debug is not None:
        debug_path = debug
        os.makedirs(debug_path, exist_ok=True)

        spectrogram_path = os.path.join(debug_path, "raw_spectrogram.pkl")
        hmm_prediction_path = os.path.join(debug_path, "hmm_predictions.pkl")
        median_prediction_path = os.path.join(debug_path, "median_predictions.pkl")

        iqr_path = os.path.join(debug_path, "iqrs.pkl")
        csv_path = os.path.join(debug_path, "classifications.csv")

        infer.save_csv_from_predictions(
            csv_path,
            hmm_predictions,
            sample_rate=spectrogram_iterator.sample_rate,
            hop_length=hop_length,
            name_to_class_code=config.name_to_class_code,
        )

        infer.pickle_tensor(spectrogram_iterator.original_spectrogram, spectrogram_path)
        infer.pickle_tensor(hmm_predictions, hmm_prediction_path)
        infer.pickle_tensor(medians, median_prediction_path)
        infer.pickle_tensor(iqr, iqr_path)
