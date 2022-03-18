from pytorch_lightning import seed_everything
from time import time

seed_everything(int(time() * 1000))

import os
import torch
import logging
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from glob import glob
from argparse import ArgumentParser

from disco.models import UNet1D, UNet1DAttn
from disco.dataset import SpectrogramDatasetMultiLabel, pad_batch
from disco.config import Config
from shopty import ShoptyConfig

log = logging.getLogger(__name__)

def train(config, hparams):
    """
    Training script.
    :param config: disco.Config() object.
    :param hparams: NameSpace containing all parameters used to train the model.
    :return: None.
    """

    train_path = os.path.join(hparams.data_path, "train", "*")
    val_path = os.path.join(hparams.data_path, "validation", "*")

    if hparams.shoptimize:
        shopty_config = ShoptyConfig()
        result_file = shopty_config.results_path
        experiment_dir = shopty_config.experiment_directory
        checkpoint_dir = shopty_config.checkpoint_directory
        checkpoint_file = shopty_config.checkpoint_file
        max_iter = shopty_config.max_iter
        min_training_unit = 1
    else:
        max_iter = hparams.epochs

    if hparams.shoptimize:
        checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            dirpath=checkpoint_dir, save_last=True, save_top_k=0, verbose=True
        )
    else:
        checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            monitor="val_loss",
            filename="{epoch}-{val_loss:.5f}-{val_acc:.5f}",
            save_top_k=1,
        )

    if hparams.shoptimize:
        logger = pl.loggers.TensorBoardLogger(experiment_dir, name="", version="")
    else:
        logger = pl.loggers.TensorBoardLogger(hparams.log_dir)

    if not len(glob(train_path)) or not len(glob(val_path)):
        raise ValueError("no files found in one of {}, {}".format(train_path, val_path))

    train_files = glob(train_path)
    val_files = glob(val_path)

    train_dataset = SpectrogramDatasetMultiLabel(
        train_files,
        config=config,
        apply_log=hparams.log,
        vertical_trim=hparams.vertical_trim,
        bootstrap_sample=hparams.bootstrap,
        mask_beginning_and_end=True,
        begin_mask=hparams.begin_mask,
        end_mask=hparams.end_mask,
    )

    val_dataset = SpectrogramDatasetMultiLabel(
        val_files,
        config=config,
        apply_log=hparams.log,
        vertical_trim=hparams.vertical_trim,
        bootstrap_sample=False,
        mask_beginning_and_end=False,
        begin_mask=None,
        end_mask=None,
    )

    # batch size of 1 since we're evaluating on lots of differently sized
    # labeled regions. Could implement a mask to zero out bits of predictions
    # but this is too much work for the amount of time it takes to evaluate our validation
    # set.
    def wrap_pad_batch(batch):
        return pad_batch(batch, config.mask_flag)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=hparams.num_workers,
        collate_fn=None if hparams.batch_size == 1 else wrap_pad_batch,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=hparams.num_workers,
        collate_fn=None if hparams.batch_size == 1 else wrap_pad_batch,
    )

    in_channels = int(config.default_spectrogram_num_rows - hparams.vertical_trim)

    model_kwargs = {
        "in_channels": in_channels,
        "learning_rate": hparams.learning_rate,
        "mel": hparams.mel,
        "apply_log": hparams.log,
        "n_fft": hparams.n_fft,
        "vertical_trim": hparams.vertical_trim,
        "mask_beginning_and_end": True,
        "begin_mask": hparams.begin_mask,
        "end_mask": hparams.end_mask,
        "train_files": train_dataset.files,
        "val_files": val_dataset.files,
        "mask_character": Config().mask_flag,
    }

    if hparams.apply_attn:
        model = UNet1DAttn(**model_kwargs)
    else:
        model = UNet1D(**model_kwargs)

    last_epoch = 0

    if hparams.shoptimize:
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            last_epoch = checkpoint["epoch"]
            model = model.load_from_checkpoint(
                checkpoint_file, map_location=torch.device("cuda")
            )

    lr_monitor = pl.callbacks.lr_monitor.LearningRateMonitor()

    trainer_kwargs = {
        "gpus": hparams.gpus,
        "num_nodes": hparams.num_nodes,
        "max_epochs": last_epoch + (max_iter * min_training_unit),
        "check_val_every_n_epoch": hparams.check_val_every_n_epoch,
        "callbacks": [checkpoint_callback, lr_monitor],
        "accelerator": "ddp" if hparams.gpus else None,
        "plugins": DDPPlugin(find_unused_parameters=False) if hparams.gpus else None,
        "precision": 16 if hparams.gpus else 32,
        "default_root_dir": hparams.log_dir,
        "log_every_n_steps": 1,
        "terminate_on_nan": True,
        "logger": logger,
    }

    if hparams.tune_initial_lr:
        trainer_kwargs["auto_lr_find"] = True

        trainer = pl.Trainer(**trainer_kwargs)

        trainer.tune(model, train_loader, val_loader)

    else:
        trainer = pl.Trainer(**trainer_kwargs)

    ckpt_path = None
    if hparams.shoptimize:
        ckpt_path = checkpoint_file if os.path.isfile(checkpoint_file) else None

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    if hparams.shoptimize:
        results = trainer.validate(model, val_loader)[0]
        log.info(results)
        with open(result_file, "w") as dst:
            dst.write(f"val_loss:{results['val_loss']}")


def main(args):
    """Run as a module."""
    train(args)
