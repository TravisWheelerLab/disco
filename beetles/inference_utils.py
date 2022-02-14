import os
import pdb
import torch
import torchaudio
import numpy as np
import pandas as pd
import pickle
import requests
import tqdm
import matplotlib.pyplot as plt
import pomegranate as pom
import logging
from glob import glob
from collections import defaultdict

from beetles.models import UNet1D
import beetles.heuristics as heuristics

log = logging.getLogger(__name__)


def create_hmm(transition_matrix, emission_probs, start_probs):
    """
    Defines the hmm to smooth post-ensemble predictions.
    :param transition_matrix: List of lists describing the hmm transition matrix.
    :param emission_probs: List of dicts.
    :param start_probs: List.
    :return: the pomegranate.HiddenMarkovModel specified by the inputs.
    """

    dists = []

    for dist in emission_probs:
        dists.append(pom.DiscreteDistribution(dist))

    transition_matrix = np.asarray(transition_matrix)
    start_probs = np.asarray(start_probs)

    hmm_model = pom.HiddenMarkovModel.from_matrix(transition_matrix, dists, start_probs)
    hmm_model.bake()

    return hmm_model


def download_models(directory, aws_download_link):
    """
    Download models from an AWS bucket.
    :param directory: Where to save the models.
    :param aws_download_link: The AWS link to download from.
    :return: None. Models are saved locally.
    """

    if not os.path.isdir(directory):
        os.makedirs(directory)

    for model_id in tqdm.tqdm(range(0, 10), desc="download status"):
        download_url = aws_download_link.format(model_id)
        download_destination = os.path.join(directory, "model_{}.pt".format(model_id))
        if not os.path.isfile(download_destination):
            f = requests.get(download_url)
            if f.status_code == 200:
                with open(download_destination, "wb") as dst:
                    dst.write(f.content)
            else:
                raise requests.RequestException(
                    f"Couldn't download model with code {f.status_code}"
                )


def aggregate_predictions(predictions):
    """
    Converts an array of predictions into a dictionary containing as keys the class index
    and as values 2-element lists containing the start of a predicted class and the end of the predicted class.

    :param predictions: np.array, Nx1, containing pointwise predictions.
    :return: Dict mapping class index to prediction start and end.
    """
    if predictions.ndim != 1:
        raise ValueError("expected array of size N, got {}".format(predictions.shape))

    diff = np.diff(predictions)  # transition regions will be nonzero
    (idx,) = diff.nonzero()
    current_class = predictions[0]
    current_idx = 0
    class_idx_to_prediction_start_and_end = []

    if len(idx) == 0:
        log.info("Only one class found after heuristics, csv will only contain one row")
        dct = {
            "class": current_class,
            "start": current_idx,
            "end": predictions.shape[-1],
        }
        class_idx_to_prediction_start_and_end.append(dct)

    else:
        for i in range(len(idx)):
            dct = {"class": current_class, "start": current_idx, "end": idx[i]}
            class_idx_to_prediction_start_and_end.append(dct)
            current_class = predictions[idx[i] + 1]
            current_idx = idx[i] + 1

    return class_idx_to_prediction_start_and_end


def convert_spectrogram_index_to_seconds(spect_idx, hop_length, sample_rate):
    """
    Converts spectrogram index back to seconds.
    :param spect_idx: Int.
    :param hop_length: Int.
    :param sample_rate: Int.
    :return:
    """
    seconds_per_hop = hop_length / sample_rate
    return spect_idx * seconds_per_hop


def pickle_tensor(data, path):
    """
    Pickles a pytorch tensor. If .numpy() isn't called, the entire computational graph is saved resulting in 100s of
    GBs of data.
    :param data: the np.array or torch.Tensor to pickle.
    :param path: Where to save the pickled input.
    :return: None.
    """

    if not isinstance(data, np.ndarray):
        data = data.numpy()

    with open(path, "wb") as dst:
        pickle.dump(data, dst)


def load_pickle(path):
    """
    Load a pickled object.
    :param path: Filename of object.
    :return: Unpickled object.
    """
    with open(path, "rb") as src:
        data = pickle.load(src)
    return data


def save_csv_from_predictions(
    output_csv_path, predictions, sample_rate, hop_length, name_to_class_code
):
    """
    Ingest a Nx1 np.array of point-wise predictions and save a .csv with
    Selection,View,Channel,Begin Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Sound_Type
    columns.
    :param output_csv_path: str. where to save the .csv of predictions.
    :param predictions: Nx1 numpy array of predictions.
    :param sample_rate: Sample rate of predicted .wav file.
    :param hop_length: Spectrogram hop length.
    :param name_to_class_code: mapping from class name to class code (ex {"A":1}).
    :return: pandas.DataFrame describing the saved csv.
    """
    # We just need to get the beginning and end of each chirp and convert those
    # to seconds.
    class_idx_to_prediction_start_end = aggregate_predictions(predictions)

    class_idx_to_prediction_start_end = heuristics.remove_a_chirps_in_between_b_chirps(
        predictions, None, name_to_class_code, return_preds=False
    )
    class_code_to_name = {v: k for k, v in name_to_class_code.items()}
    # window size by default is n_fft. hop_length is interval b/t consecutive spectrograms
    # i don't think padding is performed when the spectrogram is calculated
    list_of_dicts_for_dataframe = []
    i = 1
    for class_to_start_and_end in class_idx_to_prediction_start_end:
        # TODO: handle the case where there's only one prediction per class
        end = class_to_start_and_end["end"]
        start = class_to_start_and_end["start"]

        dataframe_dict = {
            "Selection": i,
            "View": 0,
            "Channel": 0,
            "Begin Time (s)": convert_spectrogram_index_to_seconds(
                start, hop_length=hop_length, sample_rate=sample_rate
            ),
            "End Time (s)": convert_spectrogram_index_to_seconds(
                end, hop_length=hop_length, sample_rate=sample_rate
            ),
            "Low Freq (Hz)": 0,
            "High Freq (Hz)": 0,
            "Sound_Type": class_code_to_name[class_to_start_and_end["class"]],
        }

        list_of_dicts_for_dataframe.append(dataframe_dict)
        i += 1

    df = pd.DataFrame.from_dict(list_of_dicts_for_dataframe)
    dirname = os.path.dirname(output_csv_path)

    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)

    df.to_csv(output_csv_path, index=False)

    return df


def smooth_predictions_with_hmm(unsmoothed_predictions, config):
    """
    Run the hmm defined by the config on the point-wise predictions.
    :param unsmoothed_predictions: np array of point-wise argmaxed predictions (size Nx1).
    :param config: beetles.Config() object.
    :return: smoothed predictions
    """
    if unsmoothed_predictions.ndim != 1:
        raise ValueError(
            "expected array of size N, got {}".format(unsmoothed_predictions.shape)
        )

    hmm = create_hmm(
        config.hmm_transition_probabilities,
        config.hmm_emission_probabilities,
        config.hmm_start_probabilities,
    )
    # forget about the first element b/c it's the start state
    smoothed_predictions = np.asarray(
        hmm.predict(sequence=unsmoothed_predictions.copy(), algorithm="viterbi")[1:]
    )
    return smoothed_predictions


def convert_argmaxed_array_to_rgb(predictions):
    """
    Utility function for visualization. Converts categorical labels (0, 1, 2) to RGB vectors ([1, 0, 0], [0, 1, 0],
    [0, 0, 1])
    :param predictions: Nx1 np.array.
    :return: Nx3 np.array.
    """
    rgb = np.zeros((1, predictions.shape[-1], 3))
    for class_idx in CLASS_CODE_TO_NAME.keys():
        rgb[:, np.where(predictions == class_idx), class_idx] = 1
    return rgb


def convert_time_to_spect_index(time, hop_length, sample_rate):
    """
    Converts time (seconds) to spectrogram index.
    :param time: int. Time in whatever unit your .wav file was sampled with.
    :param hop_length: Spectrogram calculation parameter.
    :param sample_rate: Sample rate of recording.
    :return: int. Spectrogram index of the time passed in.
    """

    return np.round(time * sample_rate).astype(np.int) // hop_length


def load_prediction_csv(csv_path, hop_length, sample_rate):
    """
    Utility function to load a prediction csv for visualization.
    :param csv_path: Path to the .csv containing predictions.
    :param hop_length:
    :param sample_rate:
    :return:
    """
    df = pd.read_csv(csv_path)
    df["Begin Spect Index"] = [
        convert_time_to_spect_index(x, hop_length, sample_rate)
        for x in df["Begin Time (s)"]
    ]
    df["End Spect Index"] = [
        convert_time_to_spect_index(x, hop_length, sample_rate)
        for x in df["End Time (s)"]
    ]
    return df


def assemble_ensemble(model_directory, model_extension, device, in_channels, config):
    """
    Load models in from disk or download them from an AWS bucket.
    :param model_directory: Location of saved models.
    :param model_extension: Glob extension to load the models.
    :param device: 'cuda' or 'cpu'. What device to place the models on.
    :param in_channels: How many channels the models accepts
    :param config: beetles.Config() object.
    :return: list of torch.Models().
    """

    if model_directory is None:
        model_directory = config.default_model_directory

    model_paths = glob(os.path.join(model_directory, "*" + model_extension))
    if not len(model_paths):
        log.info("no models found, downloading to {}".format(model_directory))

        download_models(config.default_model_directory)
        model_paths = glob(
            os.path.join(config.default_model_directory, "*" + model_extension)
        )

    models = []
    for model_path in model_paths:
        skeleton = UNet1D(
            in_channels,
            learning_rate=1e-2,
            mel=False,
            apply_log=False,
            n_fft=None,
            vertical_trim=20,
            mask_beginning_and_end=None,
            begin_mask=None,
            end_mask=None,
            train_files=[1],
            val_files=[1],
            mask_character=config.mask_flag,
        ).to(device)
        skeleton.load_state_dict(
            torch.load(model_path, map_location=torch.device(device))
        )
        models.append(skeleton.eval())

    return models


def load_wav_file(wav_filename):
    """
    Load a .wav file from disk.
    :param wav_filename: str. .wav file.
    :return: tuple (torch.Tensor(), int). The .wav file's data and sample rate, respectively.
    """
    waveform, sample_rate = torchaudio.load(wav_filename)
    return waveform, sample_rate


def predict_with_ensemble(ensemble, features):
    """
    Predict an array of features with a model ensemble.
    :param ensemble: List of models.
    :param features: torch.Tensor.
    :return: List of np.arrays. One for each model in the ensemble.
    """

    ensemble_preds = []

    for model in ensemble:
        preds = torch.exp(model(features))
        ensemble_preds.append(preds.to("cpu").numpy())

    return ensemble_preds


def calculate_median_and_iqr(ensemble_preds):
    """
    Get the median prediction and iqr of softmax values of the predictions from each model in the ensemble.
    :param ensemble_preds: List of np.arrays, one for each model.
    :return: tuple (np.array, np.array) of iqrs and medians.
    """

    iqrs = np.zeros(
        (ensemble_preds.shape[1], ensemble_preds.shape[2], ensemble_preds.shape[3])
    )
    medians = np.zeros(
        (ensemble_preds.shape[1], ensemble_preds.shape[2], ensemble_preds.shape[3])
    )
    for class_idx in range(ensemble_preds.shape[2]):
        q75 = np.percentile(ensemble_preds[:, :, class_idx, :], [60], axis=0)
        median = np.median(ensemble_preds[:, :, class_idx, :], axis=0)
        iqrs[:, class_idx] = q75
        medians[:, class_idx] = median

    return iqrs, medians


def evaluate_spectrogram(
    spectrogram_dataset, models, tile_overlap, original_spectrogram_shape, device="cpu"
):
    """
    Use the overlap-tile strategy to seamlessly evaluate a spectrogram.
    :param spectrogram_dataset: torch.data.DataLoader()
    :param models: list of model ensemble.
    :param tile_overlap: How much to overlap the tiles.
    :param original_spectrogram_shape: Shape of original spectrogram.
    :param device: 'cuda' or 'cpu'
    :return: medians and iqrs.
    """
    assert_accuracy = False

    with torch.no_grad():
        medians_full_sequence = []
        iqrs_full_sequence = []
        if assert_accuracy:
            all_features = []

        for features in spectrogram_dataset:

            features = features.to(device)
            ensemble_preds = predict_with_ensemble(models, features)

            if assert_accuracy:
                all_features.extend(
                    np.stack(
                        [
                            seq[:, tile_overlap:-tile_overlap]
                            for seq in features.to("cpu").numpy()
                        ]
                    )
                )

            ensemble_preds = np.stack(
                [seq[:, :, tile_overlap:-tile_overlap] for seq in ensemble_preds]
            )
            iqrs, medians = calculate_median_and_iqr(ensemble_preds)
            medians_full_sequence.extend(medians)
            iqrs_full_sequence.extend(iqrs)

    if assert_accuracy:
        all_features = np.concatenate(all_features, axis=-1)[
            :, : original_spectrogram_shape[-1]
        ]
        assert np.all(all_features == SpectrogramIterator.original_spectrogram.numpy())

    medians_full_sequence = np.concatenate(medians_full_sequence, axis=-1)[
        :, : original_spectrogram_shape[-1]
    ]
    iqrs_full_sequence = np.concatenate(iqrs_full_sequence, axis=-1)[
        :, : original_spectrogram_shape[-1]
    ]

    return medians_full_sequence, iqrs_full_sequence


class SpectrogramIterator(torch.nn.Module):
    # TODO: replace args in __init__ with sa.form_spectrogram_type
    def __init__(
        self,
        tile_size,
        tile_overlap,
        wav_file,
        vertical_trim,
        n_fft,
        hop_length,
        log_spect,
        mel_transform,
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

        waveform, self.sample_rate = load_wav_file(self.wav_file)
        self.spectrogram = self.create_spectrogram(waveform, self.sample_rate)[
            vertical_trim:
        ]
        self.original_spectrogram = self.spectrogram.clone()
        self.original_shape = self.spectrogram.shape

        step_size = self.tile_size - 2 * self.tile_overlap
        leftover = self.spectrogram.shape[-1] % step_size
        # Since the length of our spectrogram % step_size isn't always 0, we will have a little
        # leftover at the end of spectrogram that we need to predict to get full coverage. There
        # are multiple ways to do this but I decided to mirror pad the end of the spectrogram with
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
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )(waveform)
        if self.log_spect:
            spectrogram = spectrogram.log2()

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
