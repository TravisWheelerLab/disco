import logging
import os
import warnings
from glob import glob

import numpy as np
import torch

import disco.util.heuristics as heuristics
import disco.util.inference_utils as infer
from disco.datasets.dataset import SpectrogramDatasetMultiLabel

# removes torchaudio warning that spectrogram calculation needs different parameters
warnings.filterwarnings("ignore", category=UserWarning)

log = logging.getLogger(__name__)


def run_inference(
    wav_file=None,
    output_csv_path=None,
    filter_csv_label=None,
    saved_model_directory=None,
    model_extension=".pt",
    metrics_path=None,
    tile_overlap=128,
    tile_size=1024,
    batch_size=32,
    input_channels=108,
    hop_length=200,
    vertical_trim=20,
    n_fft=1150,
    viz_path=None,
    accuracy_metrics=None,
    accuracy_metrics_test_directory=None,
    num_threads=4,
    snr=0,
    name_to_class_code=None,
    add_beeps=False,
    map_unconfident=False,
    aws_download_link=None,
    class_code_to_name=None,
    mask_flag=None,
    map_to=None,
    blackout_unconfident_in_viz=False,
    default_model_directory=None,
    hmm_transition_probabilities=None,
    hmm_start_probabilities=None,
    hmm_emission_probabilities=None,
    seed=None,
):

    if tile_size % 2 != 0:
        raise ValueError("tile_size must be even, got {}".format(tile_size))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(num_threads)

    models = infer.assemble_ensemble(
        saved_model_directory,
        model_extension,
        device,
        input_channels,
        default_model_directory,
        aws_download_link,
        mask_flag,
        class_code_to_name,
    )
    if len(models) < 1:
        raise ValueError(
            "expected 1 or more models, found {}. Is model directory and extension correct?".format(
                len(models)
            )
        )
    if accuracy_metrics:
        test_files = glob(os.path.join(accuracy_metrics_test_directory, "*.pkl"))

        if not len(test_files):
            raise ValueError(
                f"Didn't find any .pkl files in {accuracy_metrics_test_directory}."
            )

        test_dataset = SpectrogramDatasetMultiLabel(
            test_files,
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

        spect, labels, iqr, medians, means, votes = infer.evaluate_pickles(
            test_loader, models, device=device
        )

    else:
        spectrogram_iterator = infer.SpectrogramIterator(
            tile_size,
            tile_overlap,
            vertical_trim=vertical_trim,
            n_fft=n_fft,
            hop_length=hop_length,
            log_spect=True,
            mel_transform=True,
            snr=snr,
            add_beeps=add_beeps,
            wav_file=wav_file,
        )
        spectrogram_dataset = torch.utils.data.DataLoader(
            spectrogram_iterator, shuffle=False, batch_size=batch_size, drop_last=False
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
            predictions = heuristic(predictions, iqr, name_to_class_code)

    if map_unconfident:
        predictions = infer.map_unconfident(
            predictions,
            to=map_to,
            threshold_type="iqr",
            threshold=1.0,
            thresholder=iqr,
        )

    hmm_predictions = infer.smooth_predictions_with_hmm(
        predictions,
        hmm_transition_probabilities,
        hmm_emission_probabilities,
        hmm_start_probabilities,
    )

    if blackout_unconfident_in_viz:
        predictions = infer.map_unconfident(
            predictions,
            to="dummy_class",
            threshold_type="iqr",
            threshold=1.0,
            thresholder=iqr,
        )

    if output_csv_path is not None:
        _, ext = os.path.splitext(output_csv_path)

        if not ext:
            output_csv_path = f"{output_csv_path}.csv"

        infer.save_csv_from_predictions(
            output_csv_path,
            hmm_predictions,
            sample_rate=spectrogram_iterator.sample_rate,
            hop_length=hop_length,
            name_to_class_code=name_to_class_code,
            filter_csv_label=filter_csv_label,
        )

    if viz_path is not None:

        spectrogram_path = os.path.join(viz_path, "raw_spectrogram.pkl")
        hmm_prediction_path = os.path.join(viz_path, "hmm_predictions.pkl")
        median_prediction_path = os.path.join(viz_path, "median_predictions.pkl")
        mean_prediction_path = os.path.join(viz_path, "mean_predictions.pkl")
        votes_path = os.path.join(viz_path, "votes.pkl")
        iqr_path = os.path.join(viz_path, "iqrs.pkl")
        csv_path = os.path.join(viz_path, "classifications.csv")

        if not accuracy_metrics:
            infer.save_csv_from_predictions(
                csv_path,
                hmm_predictions,
                sample_rate=spectrogram_iterator.sample_rate,
                hop_length=hop_length,
                name_to_class_code=name_to_class_code,
                filter_csv_label=filter_csv_label,
            )
            infer.pickle_tensor(
                spectrogram_iterator.original_spectrogram, spectrogram_path
            )
        else:
            infer.pickle_tensor(spect, spectrogram_path)

        infer.pickle_tensor(hmm_predictions, hmm_prediction_path)
        infer.pickle_tensor(predictions, median_prediction_path)
        infer.pickle_tensor(iqr, iqr_path)
        infer.pickle_tensor(means, mean_prediction_path)
        infer.pickle_tensor(votes, votes_path)

    if accuracy_metrics:
        if metrics_path is None:
            raise ValueError("Must pass a string for metrics path.")
        if not os.path.isdir(metrics_path):
            log.info(f"Creating accuracy metrics directory {metrics_path}.")
            os.makedirs(metrics_path, exist_ok=True)

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
