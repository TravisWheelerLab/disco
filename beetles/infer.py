import pdb
import warnings
import numpy as np
import os
import torch

from argparse import ArgumentParser

import beetles.heuristics as heuristics
import beetles.inference_utils as infer

# get rid of torchaudio warning us that our spectrogram calculation needs different parameters
warnings.filterwarnings("ignore", category=UserWarning)


def run_inference(
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

    if wav_file is None or output_csv_path is None:
        raise ValueError("specify both wav_file and output_csv_path")

    if tile_size % 2 != 0:
        raise ValueError("tile_size must be even, got {}".format(tile_size))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(num_threads)
    models = infer.assemble_ensemble(
        saved_model_directory, model_extension, device, input_channels
    )

    if len(models) < 2:
        raise ValueError(
            "expected more than 1 model, found {}. Is the model directory and extension correct?".format(
                len(models)
            )
        )

    spectrogram_iterator = infer.SpectrogramIterator(
        tile_size,
        tile_overlap,
        wav_file,
        vertical_trim=vertical_trim,
        n_fft=n_fft,
        hop_length=hop_length,
        log_spect=True,
        mel_transform=True,
    )

    spectrogram_dataset = torch.utils.data.DataLoader(
        spectrogram_iterator, shuffle=False, batch_size=batch_size, drop_last=False
    )

    medians, iqr = infer.evaluate_spectrogram(
        spectrogram_dataset,
        models,
        tile_overlap,
        spectrogram_iterator.original_shape,
        device=device,
    )

    predictions = np.argmax(medians, axis=0).squeeze()
    for heuristic in heuristics.HEURISTIC_FNS:
        print("applying heuristic function", heuristic.__name__)
        predictions = heuristic(predictions, iqr)

    hmm_predictions = infer.smooth_predictions_with_hmm(predictions)

    if output_csv_path is not None:
        infer.save_csv_from_predictions(
            output_csv_path,
            hmm_predictions,
            sample_rate=spectrogram_iterator.sample_rate,
            hop_length=hop_length,
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
        )

        infer.pickle_data(spectrogram_iterator.original_spectrogram, spectrogram_path)
        infer.pickle_data(hmm_predictions, hmm_prediction_path)
        infer.pickle_data(medians, median_prediction_path)
        infer.pickle_data(iqr, iqr_path)
