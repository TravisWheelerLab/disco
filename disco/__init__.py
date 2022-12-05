import logging
import os
import pdb
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from sacred.observers import FileStorageObserver

from disco.callbacks import CallbackSet
from disco.cfg import (
    extract_experiment,
    infer_experiment,
    label_experiment,
    shuffle_experiment,
    train_experiment,
    viz_experiment,
)
from disco.util.loading import load_dataset_class, load_model_class

logger = logging.getLogger(__file__)


@train_experiment.config
def _inject_semi_permanent():
    """
    Semi-permanent arguments used for the beetles dataset.
    """

    name_to_class_code = {"A": 0, "B": 1, "BACKGROUND": 2, "X": 2}
    class_code_to_name = {0: "A", 1: "B", 2: "BACKGROUND"}

    name_to_rgb_code = {"A": "#B65B47", "B": "#A36DE9", "BACKGROUND": "#AAAAAA"}
    visualization_columns = 600

    excluded_classes = ("Y", "C")

    mask_flag = -1
    default_spectrogram_num_rows = 128


@label_experiment.config
def _label_semi_permanent():
    visualization_n_fft = 1150
    vertical_cut = 20
    key_to_label = {"y": "A", "w": "B", "e": "BACKGROUND"}


@train_experiment.config
def _observer(log_dir, model_name):
    train_experiment.observers.append(FileStorageObserver(f"{log_dir}/{model_name}/"))


@train_experiment.config
def _cls_loader(model_name, dataset_name):
    model_class = load_model_class(model_name)
    dataset_class = load_dataset_class(dataset_name)


@train_experiment.main
def train(_config):
    params = SimpleNamespace(**_config)
    model = params.model_class(**params.model_args)
    train_dataset = params.dataset_class(**params.train_dataset_args)

    if hasattr(params, "val_dataset_args"):
        val_dataset = params.dataset_class(**params.val_dataset_args)
    else:
        val_dataset = None

    print(f"Training model {params.model_name} with dataset {params.dataset_name}.")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn(),
        **params.dataloader_args,
    )

    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            collate_fn=val_dataset.collate_fn(),
            **params.dataloader_args,
        )
    else:
        val_dataloader = None

    tb_logger = TensorBoardLogger(
        save_dir=os.path.split(train_experiment.observers[0].dir)[0],
        version=Path(train_experiment.observers[0].dir).name,
        name="",
    )

    if hasattr(params, "description"):
        tb_logger.experiment.add_text(
            tag="description",
            text_string=params.description,
            walltime=time.time(),
        )
    else:
        logger.info("No description of training run provided.")

    trainer = Trainer(
        **params.trainer_args,
        callbacks=CallbackSet.callbacks(),
        logger=tb_logger,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


@label_experiment.main
def label(_config):

    import matplotlib.pyplot as plt

    from disco.label import SimpleLabeler

    labeler = SimpleLabeler(
        _config["wav_file"],
        _config["output_csv_path"],
        _config["key_to_label"],
        _config["visualization_n_fft"],
        _config["vertical_cut"],
    )
    plt.show()
    labeler.show()
    labeler.save_labels()


def _infer_semi_permanent():
    default_model_directory = os.path.join(os.path.expanduser("~"), ".cache", "disco")
    model_extension = ".pt"
    tile_overlap = 128
    tile_size = 1024
    batch_size = 32
    input_channels = 108
    hop_length = 200
    vertical_trim = 20
    n_fft = 1150
    num_threads = 4
    snr = 0
    add_beeps = False
    aws_download_link = (
        "https://disco-models.s3.us-west-1.amazonaws.com/random_init_model_{}.ckpt"
    )
    map_unconfident = False
    map_to = None
    name_to_class_code = {"A": 0, "B": 1, "BACKGROUND": 2, "X": 2}
    class_code_to_name = {0: "A", 1: "B", 2: "BACKGROUND"}
    mask_flag = -1
    blackout_unconfident_in_viz = False

    hmm_transition_probabilities = [
        [0.995, 0.00000, 0.005],
        [0.0000, 0.995, 0.005],
        [0.00001, 0.00049, 0.9995],
    ]

    hmm_start_probabilities = [0, 0, 1]
    hmm_emission_probabilities = [
        {0: 0.995, 1: 0.00005, 2: 0.00495},
        {0: 0.1, 1: 0.88, 2: 0.020},
        {0: 0.05, 1: 0.05, 2: 0.9},
    ]


@infer_experiment.config
def _load_model_and_dataset(model_name, dataset_name):
    model_class = load_model_class(model_name)
    dataset = load_dataset_class(dataset_name)


@infer_experiment.config
def _dataloader_args(dataloader_args):
    dataloader_args["tile_size"] = 1024
    dataloader_args["tile_overlap"] = 128
    dataloader_args["vertical_trim"] = 20
    dataloader_args["n_fft"] = 1150
    dataloader_args["hop_length"] = 200
    dataloader_args["log_spect"] = (True,)
    dataloader_args["mel_transform"] = (True,)
    dataloader_args["snr"] = 0
    dataloader_args["add_beeps"] = False


@infer_experiment.main
def infer(_config):
    # from disco.infer import run_inference
    from disco.infer import predict_wav_file

    _config = dict(_config)
    del _config["model_name"]
    del _config["dataset_name"]
    if "saved_model_directory" not in _config:
        _config["saved_model_directory"] = cfg.default_model_directory

    dataloader_args = dict(_config["dataloader_args"])
    dataloader_args["wav_file"] = _config["wav_file"]

    dataset = _config["dataset"](**dataloader_args)

    del _config["dataloader_args"]
    _config["dataset"] = dataset

    predict_wav_file(**_config)


@extract_experiment.config
def _extract_semi_permanent():
    seed = 0
    no_mel_scale = False
    n_fft = 1150
    overwrite = False
    snr = 0
    add_beeps = False
    extract_context = False
    class_code_to_name = {0: "A", 1: "B", 2: "BACKGROUND"}
    name_to_class_code = {"A": 0, "B": 1, "BACKGROUND": 2, "X": 2}
    excluded_classes = ("Y", "C")


@extract_experiment.main
def extract(_config):
    from disco.util.extract_data import extract_single_file

    extract_single_file(**_config)


@viz_experiment.config
def viz_semi_permanent():

    class_code_to_name = {0: "A", 1: "B", 2: "BACKGROUND"}
    name_to_rgb_code = {"A": "#B65B47", "B": "#A36DE9", "BACKGROUND": "#AAAAAA"}
    visualization_columns = 600


@viz_experiment.main
def visualize(_config):
    from disco.visualize import visualize as viz

    viz(**_config)


@shuffle_experiment.config
def shuffle_semi_permanent():

    train_pct = 0.8
    extension = ".pkl"


@shuffle_experiment.main
def shuffle(_config):
    from disco.util.extract_data import shuffle_data

    shuffle_data(**_config)


def main():

    if sys.argv[1] == "train":
        train_experiment.run_commandline(sys.argv[1:])
    elif sys.argv[1] == "label":
        label_experiment.run_commandline(sys.argv[1:])
    elif sys.argv[1] == "infer":
        infer_experiment.run_commandline(sys.argv[1:])
    elif sys.argv[1] == "extract":
        extract_experiment.run_commandline(sys.argv[1:])
    elif sys.argv[1] == "viz":
        viz_experiment.run_commandline(sys.argv[1:])
    elif sys.argv[1] == "shuffle":
        shuffle_experiment.run_commandline(sys.argv[1:])
    else:
        raise ValueError(
            "must choose one of <train, label, infer, extract, viz, shuffle>"
        )


if __name__ == "__main__":
    main()
