import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import pickle
import requests
import tqdm
import pomegranate as pom
import logging
from glob import glob
from disco.accuracy_metrics import adjust_preds_by_confidence
import pdb

from disco.models import UNet1D
import disco.heuristics as heuristics

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
                raise requests.RequestException(f"Couldn't download model with code {f.status_code}")


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
        dct = {"class": current_class, "start": current_idx, "end": predictions.shape[-1]}
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
    :param spect_idx: Int. of spectrogram index
    :param hop_length: Int. of hop length
    :param sample_rate: Int. of wav file sample rate
    :return: int of converted seconds.
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
        output_csv_path, predictions, sample_rate, hop_length, name_to_class_code, noise_pct, filter_csv_label,
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
    :param noise_pct: Float of how much noise is added to the spectrogram.
    :param filter_csv_label: str. remove a class from final csv.
    :return: pandas.DataFrame describing the saved csv.
    """
    class_idx_to_prediction_start_end = heuristics.remove_a_chirps_in_between_b_chirps(
        predictions,
        None,
        name_to_class_code,
        return_preds=False
    )
    class_code_to_name = {v: k for k, v in name_to_class_code.items()}
    list_of_dicts_for_dataframe = []
    i = 1
    for class_to_start_and_end in class_idx_to_prediction_start_end:
        end = class_to_start_and_end["end"]
        start = class_to_start_and_end["start"]

        dataframe_dict = {
            "Selection": i,
            "View": 0,
            "Channel": 0,
            "Begin Time (s)": convert_spectrogram_index_to_seconds(start, hop_length=hop_length, sample_rate=sample_rate),
            "End Time (s)": convert_spectrogram_index_to_seconds(end, hop_length=hop_length, sample_rate=sample_rate),
            "Low Freq (Hz)": 0,
            "High Freq (Hz)": 0,
            "Sound_Type": class_code_to_name[class_to_start_and_end["class"]]
        }

        list_of_dicts_for_dataframe.append(dataframe_dict)
        i += 1

    df = pd.DataFrame.from_dict(list_of_dicts_for_dataframe)

    if filter_csv_label:
        df = df[df['Sound_Type'] != filter_csv_label]

    dirname = os.path.dirname(output_csv_path)

    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)

    if noise_pct > 0:
        output_csv_path = output_csv_path + "_" + str(noise_pct) + "_pctnoise"

    df.to_csv(output_csv_path, index=False)

    return df


def smooth_predictions_with_hmm(unsmoothed_predictions, config):
    """
    Run the hmm defined by the config on the point-wise predictions.
    :param unsmoothed_predictions: np array of point-wise argmaxed predictions (size Nx1).
    :param config: disco.Config() object.
    :return: smoothed predictions
    """
    if unsmoothed_predictions.ndim != 1:
        raise ValueError("expected array of size N, got {}".format(unsmoothed_predictions.shape))

    hmm = create_hmm(
        config.hmm_transition_probabilities,
        config.hmm_emission_probabilities,
        config.hmm_start_probabilities,
    )
    # forget about the first element because it's the start state
    smoothed_predictions = np.asarray(hmm.predict(sequence=unsmoothed_predictions.copy(), algorithm="viterbi")[1:])
    return smoothed_predictions


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
    :return: original dataframe but with new columns, "Begin Spect Index" and "End Spect Index".
    """
    df = pd.read_csv(csv_path)
    df["Begin Spect Index"] = [convert_time_to_spect_index(x, hop_length, sample_rate) for x in df["Begin Time (s)"]]
    df["End Spect Index"] = [convert_time_to_spect_index(x, hop_length, sample_rate) for x in df["End Time (s)"]]
    return df


def assemble_ensemble(model_directory, model_extension, device, in_channels, config):
    """
    Load models in from disk or download them from an AWS bucket.
    :param model_directory: Location of saved models.
    :param model_extension: Glob extension to load the models.
    :param device: 'cuda' or 'cpu'. What device to place the models on.
    :param in_channels: How many channels the models accepts
    :param config: disco.Config() object.
    :return: list of torch.Models().
    """

    if model_directory is None:
        model_directory = config.default_model_directory
    else:
        model_paths = glob(os.path.join("disco", model_directory, "*", "*", "*" + model_extension), recursive=True)
    if not len(model_paths):
        log.info("no models found, downloading to {}".format(model_directory))

        download_models(config.default_model_directory, config.aws_download_link)
        model_paths = glob(os.path.join(model_directory, "*"))

    models = []
    for model_path in model_paths:
        skeleton = UNet1D(
            in_channels,
            out_channels=len(config.class_code_to_name.keys()),
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

        if model_extension == ".ckpt":
            skeleton = skeleton.load_from_checkpoint(model_path)
        elif model_extension == ".pt":
            skeleton.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        else:
            raise RuntimeError("Unrecognized model extension.")

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

@torch.no_grad()
def predict_with_ensemble(ensemble, features):
    """
    Predict an array of features with a model ensemble.
    :param ensemble: List of models.
    :param features: torch.Tensor.
    :return: List of np.arrays. One for each model in the ensemble.
    """

    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"

    ensemble_preds = []

    for model in ensemble:
        model = model.to(dev)
        preds = torch.nn.functional.softmax((model(features.to(dev))), dim=1)
        ensemble_preds.append(preds.to("cpu").numpy())

    return ensemble_preds


def calculate_ensemble_statistics(ensemble_preds):
    """
    Get the median prediction and iqr of softmax values of the predictions from each model in the ensemble.
    :param ensemble_preds: List of np.arrays, one for each model.
    :return: tuple (np.array, np.array) of iqrs and medians.
    """

    number_of_models = ensemble_preds.shape[0]
    number_of_spectrograms = ensemble_preds.shape[1]
    number_of_classes = ensemble_preds.shape[2]
    length_per_spectrogram = ensemble_preds.shape[3]

    iqrs = np.zeros((number_of_spectrograms, number_of_classes, length_per_spectrogram))
    medians = np.zeros((number_of_spectrograms, number_of_classes, length_per_spectrogram))
    means = np.zeros((number_of_spectrograms, number_of_classes, length_per_spectrogram))
    votes = np.zeros((number_of_spectrograms, number_of_classes, length_per_spectrogram))

    for class_idx in range(number_of_classes):
        q75, q25 = np.percentile(ensemble_preds[:, :, class_idx, :], [75, 25], axis=0)
        median = np.median(ensemble_preds[:, :, class_idx, :], axis=0)
        mean = np.mean(ensemble_preds[:, :, class_idx, :], axis=0)

        iqrs[:, class_idx] = q75 - q25
        medians[:, class_idx] = median
        means[:, class_idx] = mean

    for ensemble_member in range(number_of_models):
        for spectrogram_number in range(number_of_spectrograms):
            slice_to_analyze = ensemble_preds[ensemble_member, spectrogram_number, :, :]
            class_votes = list(np.argmax(slice_to_analyze, axis=0))
            for i in range(len(class_votes)):
                votes[spectrogram_number, class_votes[i], i] += 1

    return iqrs, medians, means, votes


def evaluate_spectrogram(
        spectrogram_dataset, models, tile_overlap, original_spectrogram, original_spectrogram_shape, device="cpu"
):
    """
    Use the overlap-tile strategy to seamlessly evaluate a spectrogram.
    :param spectrogram_dataset: torch.data.DataLoader()
    :param models: list of model ensemble.
    :param tile_overlap: How much to overlap the tiles.
    :param original_spectrogram: Original spectrogram.
    :param original_spectrogram_shape: Shape of original spectrogram.
    :param device: 'cuda' or 'cpu'
    :return: medians, iqrs, means, and votes, each numpy arrays that have a shape of (classes, length).
    """
    assert_accuracy = True

    with torch.no_grad():
        iqrs_full_sequence = []
        medians_full_sequence = []
        means_full_sequence = []
        votes_full_sequence = []
        if assert_accuracy:
            all_features = []

        for features in spectrogram_dataset:
            features = features.to(device)
            ensemble_preds = predict_with_ensemble(models, features)
            if assert_accuracy:
                all_features.extend(np.stack([seq[:, tile_overlap:-tile_overlap]
                                              for seq in features.to("cpu").numpy()]))

            ensemble_preds = np.stack([seq[:, :, tile_overlap:-tile_overlap]
                                       for seq in ensemble_preds])
            iqrs, medians, means, votes = calculate_ensemble_statistics(ensemble_preds)

            iqrs_full_sequence.extend(iqrs)
            medians_full_sequence.extend(medians)
            means_full_sequence.extend(means)
            votes_full_sequence.extend(votes)

    if assert_accuracy:
        all_features = np.concatenate(all_features, axis=-1)[:, : original_spectrogram_shape[-1]]
        assert np.all(all_features == original_spectrogram.numpy())

    iqrs_full_sequence = np.concatenate(iqrs_full_sequence, axis=-1)[:, :original_spectrogram_shape[-1]]
    medians_full_sequence = np.concatenate(medians_full_sequence, axis=-1)[:, :original_spectrogram_shape[-1]]
    means_full_sequence = np.concatenate(means_full_sequence, axis=-1)[:, :original_spectrogram_shape[-1]]
    votes_full_sequence = np.concatenate(votes_full_sequence, axis=-1)[:, :original_spectrogram_shape[-1]]

    return iqrs_full_sequence, medians_full_sequence, means_full_sequence, votes_full_sequence


def map_unconfident(predictions, to, threshold_type, threshold, thresholder, config):
    if threshold_type == "iqr":
        thresholder = np.mean(thresholder, axis=0)
    else:
        "Not currently set up to do any unconfidence mappings other than iqr."
    if to == "dummy_class":
        unconfident_class = max(config.class_code_to_name.keys()) + 1
    elif to == "BACKGROUND":
        unconfident_class = config.name_to_class_code["BACKGROUND"]
    elif to is None:
        unconfident_class = max(config.class_code_to_name.keys())

    predictions = adjust_preds_by_confidence(average_iqr=thresholder, iqr_threshold=threshold, winning_vote_count=thresholder, min_votes_needed=-1, median_argmax=predictions, map_to=unconfident_class)
    return predictions


def evaluate_pickles(
        spectrogram_dataset, models, device="cpu"
):
    """
    Use the overlap-tile strategy to seamlessly evaluate a spectrogram.
    :param spectrogram_dataset: torch.data.DataLoader()
    :param models: list of model ensemble.
    :param device: 'cuda' or 'cpu'
    :return: medians, iqrs, means, and votes, each numpy arrays that have a shape of (classes, length).
    """
    with torch.no_grad():
        spectrograms_concat = []
        labels_concat = []
        iqrs_full_sequence = []
        medians_full_sequence = []
        means_full_sequence = []
        votes_full_sequence = []

        for features, labels in spectrogram_dataset:
            features = features.to(device)
            ensemble_preds = predict_with_ensemble(models, features)

            ensemble_preds = np.stack(ensemble_preds)
            iqrs, medians, means, votes = calculate_ensemble_statistics(ensemble_preds)

            spectrograms_concat.extend(features.to("cpu"))
            labels_concat.extend(labels.to("cpu"))
            iqrs_full_sequence.extend(iqrs)
            medians_full_sequence.extend(medians)
            means_full_sequence.extend(means)
            votes_full_sequence.extend(votes)

    spectrograms_concat = np.concatenate(spectrograms_concat, axis=-1)
    labels_concat = np.concatenate(labels_concat, axis=-1)
    iqrs_full_sequence = np.concatenate(iqrs_full_sequence, axis=-1)
    medians_full_sequence = np.concatenate(medians_full_sequence, axis=-1)
    means_full_sequence = np.concatenate(means_full_sequence, axis=-1)
    votes_full_sequence = np.concatenate(votes_full_sequence, axis=-1)

    return spectrograms_concat, labels_concat, iqrs_full_sequence, medians_full_sequence, means_full_sequence, votes_full_sequence


def make_viz_directory(wav_file, saved_model_directory, debug_path, accuracy_metrics):
    """
    Creates a directory based on the parser's debug path and returns the updated debug path later used for saving the
    statistics. If given debug path already exists, creates a directory inside with the default name. If debug path
    doesn't already exist, creates a directory with the name provided. If there is no debug path provided, creates a
    default-named directory within the current directory.
    :param wav_file: String. filepath of the given .wav file.
    :param saved_model_directory: String. filepath of the saved models.
    :param debug_path: String. Path indicating where to save the visualization files.
    :param accuracy_metrics: bool. Whether the input files are accuracy metric (test) files.
    :return: Updated debug path used
    """
    if not accuracy_metrics:
        default_dirname = os.path.split(wav_file)[-1].split(".")[0] + "-" + os.path.split(saved_model_directory)[-1]
    else:
        default_dirname = os.path.basename(saved_model_directory) + "_test_files_visualization"

    if debug_path is not None:
        if os.path.exists(debug_path):
            debug_path = os.path.join(debug_path, default_dirname)
            os.makedirs(debug_path, exist_ok=True)
        else:
            os.makedirs(debug_path, exist_ok=True)
    else:
        debug_path = default_dirname
        os.makedirs(debug_path, exist_ok=True)
    print("Created visualizations directory: " + debug_path + ".")
    return debug_path


def add_gaussian_noise(spectrogram, noise_pct):
    """
    Adds gaussian noise to the spectrogram to experiment with out-of-distribution data. Find the standard deviation of
    spectrogram values, multiplies by the percent provided, and multiplies onto a random number torch tensor. Finally,
    adds these values to given spectrogram.
    :param spectrogram: torch tensor of spectrogram
    :param noise_pct: int. of desired noise percent of spectrogram standard deviation. This is multiplied by a random
    variable distributed as a Normal(0,1) for each pixel of the spectrogram.
    :return: Noised spectrogram
    """
    spect_sd = torch.std(spectrogram)
    noise = .01 * noise_pct
    spectrogram = spectrogram + noise * spect_sd * torch.randn(size=spectrogram.shape)
    return spectrogram


class SpectrogramIterator(torch.nn.Module):
    def __init__(
            self,
            tile_size,
            tile_overlap,
            vertical_trim,
            n_fft,
            hop_length,
            log_spect,
            mel_transform,
            noise_pct,
            wav_file=None,
            spectrogram=None,
    ):
        super().__init__()
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        if self.tile_size <= tile_overlap:
            raise ValueError()
        self.wav_file = wav_file
        self.spectrogram = spectrogram
        if self.spectrogram is None and self.wav_file is None:
            raise ValueError()
        self.vertical_trim = vertical_trim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.log_spect = log_spect
        self.mel_transform = mel_transform
        self.noise_pct = noise_pct
        if self.spectrogram is None:
            waveform, self.sample_rate = load_wav_file(self.wav_file)
            self.spectrogram = self.create_spectrogram(waveform, self.sample_rate)
        self.spectrogram = self.spectrogram[vertical_trim:]
        if not torch.is_tensor(self.spectrogram):
            self.spectrogram = torch.tensor(self.spectrogram)
        if self.noise_pct != 0:
            self.spectrogram = add_gaussian_noise(self.spectrogram, self.noise_pct)
        if self.log_spect:
            self.spectrogram[self.spectrogram == 0] = 1
            self.spectrogram = self.spectrogram.log2()

        self.original_spectrogram = self.spectrogram.clone()
        self.original_shape = self.spectrogram.shape

        step_size = self.tile_size - 2 * self.tile_overlap
        leftover = self.spectrogram.shape[-1] % step_size
        # Since the length of our spectrogram % step_size isn't always 0, we will have a little
        # leftover at the end of spectrogram that we need to predict to get full coverage. There
        # are multiple ways to do this, but I decided to mirror pad the end of the spectrogram with
        # the correct amount of columns from the spectrogram so that padded_spectrogram % step_size == 0.
        # I cut off the predictions on the mirrored data after stitching the predictions together.
        to_pad = step_size - leftover + tile_size // 2

        if to_pad != 0:
            self.spectrogram = torch.cat(
                (self.spectrogram, torch.flip(self.spectrogram[:, -to_pad:], dims=[-1])),
                dim=-1
            )

        self.indices = range(self.tile_size // 2, self.spectrogram.shape[-1], step_size)

        # mirror pad the beginning of the spectrogram
        self.spectrogram = torch.cat((torch.flip(self.spectrogram[:, : self.tile_overlap], dims=[-1]),
                                      self.spectrogram),
                                     dim=-1)

    def create_spectrogram(self, waveform, sample_rate):
        if self.mel_transform:
            spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )(waveform)
        else:
            spectrogram = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )(waveform)
        return spectrogram.squeeze()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        center_idx = self.indices[idx]
        # we want to overlap-tile starting from the beginning
        # so that our predictions are seamless.
        x = self.spectrogram[:, center_idx - self.tile_size // 2: center_idx + self.tile_size // 2]
        return x
