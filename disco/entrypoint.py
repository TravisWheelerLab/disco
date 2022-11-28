import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from sacred.observers import FileStorageObserver

from disco.callbacks import CallbackSet
from disco.config import main_experiment
from disco.util.loading import load_dataset_class, load_model_class

logger = logging.getLogger(__file__)


def _inject_semi_permanent(config):

    # TODO: fixme
    """
    Semi-permanent arguments used for the beetles dataset.
    """

    name_to_class_code = {"A": 0, "B": 1, "BACKGROUND": 2, "X": 2}
    class_code_to_name = {0: "A", 1: "B", 2: "BACKGROUND"}

    name_to_rgb_code = {"A": "#B65B47", "B": "#A36DE9", "BACKGROUND": "#AAAAAA"}
    visualization_n_fft = 1150
    visualization_columns = 600

    excluded_classes = ("Y", "C")

    default_model_directory = os.path.join(os.path.expanduser("~"), ".cache", "disco")

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

    aws_download_link = (
        "https://disco-models.s3.us-west-1.amazonaws.com/random_init_model_{}.ckpt"
    )

    vertical_cut = 20
    key_to_label = {"y": "A", "w": "B", "e": "BACKGROUND"}

    mask_flag = -1
    default_spectrogram_num_rows = 128


@main_experiment.config
def _observer(log_dir, model_name):
    main_experiment.observers.append(FileStorageObserver(f"{log_dir}/{model_name}/"))


@main_experiment.config
def _cls_loader(model_name, dataset_name):
    model_class = load_model_class(model_name)
    dataset_class = load_dataset_class(dataset_name)


@main_experiment.main
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
        save_dir=os.path.split(main_experiment.observers[0].dir)[0],
        version=Path(main_experiment.observers[0].dir).name,
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


def train_main():
    main_experiment.run_commandline()
