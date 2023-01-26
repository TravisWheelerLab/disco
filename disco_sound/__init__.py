"""
DISCO Implements Sound Classification Obediently.
"""
__version__ = "0.0.1"
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sacred.observers import FileStorageObserver

import disco_sound.cfg as cfg
from disco_sound.callbacks import CallbackSet
from disco_sound.cfg.extract_config import extract_experiment
from disco_sound.cfg.infer_config import infer_experiment
from disco_sound.cfg.label_config import label_experiment
from disco_sound.cfg.shuffle_config import shuffle_experiment
from disco_sound.cfg.train_config import train_experiment
from disco_sound.cfg.viz_config import viz_experiment
from disco_sound.util.loading import load_dataset_class, load_model_class

root = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger("disco")


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

    logger.info(
        f"Training model {params.model_name} with dataset {params.dataset_name}."
    )
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

    from disco_sound.label import SimpleLabeler

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


@infer_experiment.config
def _load_model_and_dataset(model_name, dataset_name):
    model_class = load_model_class(model_name)
    dataset = load_dataset_class(dataset_name)


@infer_experiment.main
def infer(_config):

    from disco_sound.infer import predict_wav_file

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


@extract_experiment.main
def extract(_config):
    from disco_sound.util.extract_data import extract_single_file

    extract_single_file(**_config)


@viz_experiment.main
def visualize(_config):
    from disco_sound.visualize import visualize as viz

    viz(**_config)


@shuffle_experiment.main
def shuffle(_config):
    from disco_sound.util.extract_data import shuffle_data

    shuffle_data(**_config)


def main():
    if len(sys.argv) == 1:
        logger.info(
            f"DISCO version {__version__}. Usage: "
            f"disco <label, extract, shuffle, train, infer>. "
            f"See docs at https://github.com/TravisWheelerLab/disco/wiki for more help."
        )
        exit()

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
