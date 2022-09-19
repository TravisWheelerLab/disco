import pdb
import warnings
import numpy as np
import os
import logging
import torch
from glob import glob

import disco.heuristics as heuristics
import disco.inference_utils as infer
from disco.dataset import SpectrogramDatasetMultiLabel

# removes torchaudio warning that spectrogram calculation needs different parameters
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
    viz=None,
    viz_path=None,
    accuracy_metrics=None,
    accuracy_metrics_test_directory=None,
    num_threads=4,
    noise_pct=0,
):
    """
    Script to run the inference routine. Briefly: Model ensemble is loaded in, used to evaluate the spectrogram, and
    heuristics and the hmm are applied to the ensemble's predictions. The .csv containing model labels is saved and
    debugging information is saved depending on whether debug is a string or None.

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
    :param viz: bool. Whether to save statistics of the output predictions.
    :param viz_path: str. Where to save the visualization data.  If debug path already exists, create a directory inside
    with the default name. If debug path doesn't already exist, creates a directory with the name provided. Default:
    creates a default-named directory within the current directory.
    :param accuracy_metrics: bool. whether to save a directory containing needed files to determine ensemble accuracy
    :param accuracy_metrics_test_directory: str. where test files are located.
    :param num_threads: How many threads to use when loading data.
    :param noise_pct: How much noise to add to the data.
    :return: None. Everything relevant is saved to disk.
    """

    if tile_size % 2 != 0:
        raise ValueError("tile_size must be even, got {}".format(tile_size))
    if noise_pct < 0 or noise_pct > 100:
        raise ValueError("noise_pct must be a percentage, got {}".format(noise_pct))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(num_threads)
    models = infer.assemble_ensemble(saved_model_directory, model_extension, device, input_channels, config)
    if len(models) < 1:
        raise ValueError("expected 1 or more models, found {}. Is model directory and extension correct?".format(
                len(models)
        ))
    if accuracy_metrics:
        test_files = glob(os.path.join(accuracy_metrics_test_directory, "*.pkl"))
        test_dataset = SpectrogramDatasetMultiLabel(
            test_files,
            config=config,
            apply_log=True,
            vertical_trim=20,
            bootstrap_sample=False,
            mask_beginning_and_end=False,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=None,
        )

        spect, labels, iqr, medians, means, votes = infer.evaluate_pickles(test_loader, models, device=device)

    else:
        spectrogram_iterator = infer.SpectrogramIterator(
            tile_size,
            tile_overlap,
            vertical_trim=vertical_trim,
            n_fft=n_fft,
            hop_length=hop_length,
            log_spect=True,
            mel_transform=True,
            noise_pct=noise_pct,
            wav_file=wav_file,
        )
        spectrogram_dataset = torch.utils.data.DataLoader(
            spectrogram_iterator,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False
        )
        original_spectrogram = spectrogram_iterator.original_spectrogram
        original_shape = spectrogram_iterator.original_shape

        iqr, medians, means, votes = infer.evaluate_spectrogram(
            spectrogram_dataset,
            models,
            tile_overlap,
            original_spectrogram,
            original_shape,
            device=device,
        )

    predictions = np.argmax(medians, axis=0).squeeze()

    if not accuracy_metrics:
        for heuristic in heuristics.HEURISTIC_FNS:
            log.info(f"applying heuristic function {heuristic.__name__}")
            predictions = heuristic(predictions, iqr, config.name_to_class_code)

    blackout_unconfident = True
    background_unconfident = True
    threshold_type = "iqr"
    if threshold_type == "iqr":
        thresholder = iqr
    threshold = 0.05

    # regular use case of confidence heuristics!
    if background_unconfident:
        predictions = infer.map_unconfident(predictions, to="BACKGROUND", threshold_type=threshold_type, threshold=threshold, thresholder=thresholder, config=config)

    hmm_predictions = infer.smooth_predictions_with_hmm(predictions, config)

    # only used when evaluating accuracy metrics visually!
    if blackout_unconfident:
        predictions = infer.map_unconfident(predictions, to="dummy_class", threshold_type=threshold_type, threshold=threshold, thresholder=thresholder, config=config)

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
            noise_pct=noise_pct,
        )

    if viz:
        debug_path = infer.make_viz_directory(wav_file, saved_model_directory, viz_path, accuracy_metrics)

        spectrogram_path = os.path.join(debug_path, "raw_spectrogram.pkl")
        hmm_prediction_path = os.path.join(debug_path, "hmm_predictions.pkl")
        median_prediction_path = os.path.join(debug_path, "median_predictions.pkl")
        mean_prediction_path = os.path.join(debug_path, "mean_predictions.pkl")
        votes_path = os.path.join(debug_path, "votes.pkl")
        iqr_path = os.path.join(debug_path, "iqrs.pkl")
        csv_path = os.path.join(debug_path, "classifications.csv")

        if not accuracy_metrics:
            infer.save_csv_from_predictions(
                csv_path,
                hmm_predictions,
                sample_rate=spectrogram_iterator.sample_rate,
                hop_length=hop_length,
                name_to_class_code=config.name_to_class_code,
                noise_pct=noise_pct,
            )
            infer.pickle_tensor(spectrogram_iterator.original_spectrogram, spectrogram_path)
        else:
            infer.pickle_tensor(spect, spectrogram_path)

        infer.pickle_tensor(hmm_predictions, hmm_prediction_path)
        infer.pickle_tensor(predictions, median_prediction_path)
        infer.pickle_tensor(iqr, iqr_path)
        infer.pickle_tensor(means, mean_prediction_path)
        infer.pickle_tensor(votes, votes_path)

    if accuracy_metrics:
        default_dirname = f"test_files-{os.path.basename(saved_model_directory)}"
        metrics_path = os.path.join("data", "accuracy_metrics", default_dirname)
        os.makedirs(metrics_path, exist_ok=True)

        print("Created accuracy metrics directory: " + metrics_path + ".")

        labels_path = os.path.join(metrics_path, "ground_truth.pkl")
        hmm_prediction_path = os.path.join(metrics_path, "hmm_predictions.pkl")
        median_prediction_path = os.path.join(metrics_path, "median_predictions.pkl")
        iqr_path = os.path.join(metrics_path, "iqrs.pkl")
        votes_path = os.path.join(metrics_path, "votes.pkl")

        infer.pickle_tensor(labels, labels_path)
        infer.pickle_tensor(hmm_predictions, hmm_prediction_path)
        infer.pickle_tensor(medians, median_prediction_path)
        infer.pickle_tensor(iqr, iqr_path)
        infer.pickle_tensor(votes, votes_path)