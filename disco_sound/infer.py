import logging
import os.path
import warnings

import numpy as np
import torch

import disco_sound.cfg as cfg
import disco_sound.util.inference_utils as infer

# removes torchaudio warning that spectrogram calculation needs different parameters
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


def predict_wav_file(
    wav_file,
    dataset,
    model_class,
    saved_model_directory,
    output_directory=None,
    tile_overlap=128,
    tile_size=1024,
    batch_size=32,
    hop_length=200,
    num_threads=4,
    seed=None,
):
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

    if saved_model_directory is not None:
        log.info(f"Using {len(models)} models from {saved_model_directory}.")

    if len(models) < 1:
        raise ValueError(
            "expected 1 or more models, found {}. Is model directory correct?".format(
                len(models)
            )
        )

    spectrogram_dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=batch_size, drop_last=False
    )

    original_spectrogram = dataset.original_spectrogram
    original_shape = dataset.original_shape

    iqr, medians, means, votes, preds = infer.evaluate_spectrogram(
        spectrogram_dataloader,
        models,
        tile_overlap,
        original_spectrogram,
        original_shape,
        device=device,
    )

    predictions = np.argmax(medians, axis=0).squeeze()

    hmm_predictions = infer.smooth_predictions_with_hmm(
        predictions,
        cfg.hmm_transition_probabilities,
        cfg.hmm_emission_probabilities,
        cfg.hmm_start_probabilities,
    )

    # auto-generate a directory
    if output_directory is None:
        wav_root = os.path.dirname(wav_file)
    else:
        wav_root = output_directory

    output_csv_path = os.path.join(
        wav_root, os.path.splitext(os.path.basename(wav_file))[0] + "-detected.csv"
    )

    infer.save_csv_from_predictions(
        output_csv_path,
        hmm_predictions,
        sample_rate=dataset.sample_rate,
        hop_length=hop_length,
        name_to_class_code=cfg.name_to_class_code,
    )

    # now make a visualization path
    if output_directory is None:
        viz_root = os.path.dirname(wav_file)
    else:
        viz_root = output_directory

    wav_basename = os.path.splitext(os.path.basename(wav_file))[0]
    viz_path = os.path.join(viz_root, wav_basename + "-viz")

    if os.path.isdir(viz_path):
        logger.info(f"Directory {viz_path} already exists. Not overwriting.")
        exit()
    else:
        os.makedirs(viz_path)

    spectrogram_path = os.path.join(viz_path, "raw_spectrogram.pkl")
    hmm_prediction_path = os.path.join(viz_path, "hmm_predictions.pkl")
    median_prediction_path = os.path.join(viz_path, "median_predictions.pkl")
    mean_prediction_path = os.path.join(viz_path, "mean_predictions.pkl")
    raw_pred_path = os.path.join(viz_path, "raw_preds.pkl")
    votes_path = os.path.join(viz_path, "votes.pkl")
    iqr_path = os.path.join(viz_path, "iqrs.pkl")
    csv_path = os.path.join(viz_path, "classifications.csv")

    infer.save_csv_from_predictions(
        csv_path,
        hmm_predictions,
        sample_rate=dataset.sample_rate,
        hop_length=hop_length,
        name_to_class_code=cfg.name_to_class_code,
    )

    infer.pickle_tensor(dataset.original_spectrogram, spectrogram_path)
    infer.pickle_tensor(hmm_predictions, hmm_prediction_path)
    infer.pickle_tensor(predictions, median_prediction_path)
    infer.pickle_tensor(iqr, iqr_path)
    infer.pickle_tensor(means, mean_prediction_path)
    infer.pickle_tensor(preds, raw_pred_path)
    infer.pickle_tensor(votes, votes_path)
