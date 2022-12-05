import os
from glob import glob

import numpy as np
import torch
from sacred import Experiment

import disco.cfg as cfg
import disco.util.inference_utils as infer
from disco.datasets.dataset import SpectrogramDatasetMultiLabel
from disco.util.loading import load_model_class

experiment = Experiment()


@experiment.automain
def evaluate_test_files(
    test_path,
    model_name,
    metrics_path,
    saved_model_directory=None,
    tile_size=1024,
    num_threads=4,
):

    test_files = glob(os.path.join(test_path, "*pkl"))
    model_class = load_model_class(model_name)

    if not len(test_files):
        raise ValueError("no test files.")

    if tile_size % 2 != 0:
        raise ValueError("tile_size must be even, got {}".format(tile_size))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(num_threads)

    models = infer.assemble_ensemble(
        model_class,
        saved_model_directory,
        device,
        default_model_directory=cfg.default_model_directory,
        aws_download_link=cfg.aws_download_link,
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

    spect, labels, iqr, medians, means, votes = infer.evaluate_test_loader(
        test_loader, models, device=device
    )

    predictions = np.argmax(medians, axis=0).squeeze()

    hmm_predictions = infer.smooth_predictions_with_hmm(
        predictions,
        cfg.hmm_transition_probabilities,
        cfg.hmm_emission_probabilities,
        cfg.hmm_start_probabilities,
    )

    os.makedirs(metrics_path, exist_ok=True)

    labels_path = os.path.join(metrics_path, "ground_truth.pkl")
    mean_path = os.path.join(metrics_path, "hmm_predictions.pkl")
    hmm_prediction_path = os.path.join(metrics_path, "mean_predictions.pkl")
    median_prediction_path = os.path.join(metrics_path, "median_predictions.pkl")
    iqr_path = os.path.join(metrics_path, "iqrs.pkl")
    votes_path = os.path.join(metrics_path, "votes.pkl")
    spectrogram_path = os.path.join(metrics_path, "spectrogram.pkl")

    infer.pickle_tensor(labels, labels_path)
    infer.pickle_tensor(hmm_predictions, hmm_prediction_path)
    infer.pickle_tensor(means, mean_path)
    infer.pickle_tensor(medians, median_prediction_path)
    infer.pickle_tensor(iqr, iqr_path)
    infer.pickle_tensor(votes, votes_path)
    infer.pickle_tensor(spect, spectrogram_path)
