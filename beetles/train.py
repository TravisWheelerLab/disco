from pytorch_lightning import seed_everything
from time import time

# BE CAREFUL. This will cause inconsistent behavior if training
# using DDP!
seed_everything(int(time() * 1000))

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from glob import glob
from argparse import ArgumentParser

from beetles.models import SimpleCNN, UNet1D, UNet1DAttn
from beetles.dataset import SpectrogramDatasetMultiLabel, pad_batch
from beetles import DEFAULT_SPECTROGRAM_NUM_ROWS

__all__ = ["train"]


def train(hparams):

    train_path = os.path.join(hparams.data_path, "train", "*")
    val_path = os.path.join(hparams.data_path, "validation", "*")

    if not len(glob(train_path)) or not len(glob(val_path)):
        raise ValueError("no files found in one of {}, {}".format(train_path, val_path))

    train_files = glob(train_path)
    val_files = glob(val_path)

    train_dataset = SpectrogramDatasetMultiLabel(
        train_files,
        apply_log=hparams.log,
        vertical_trim=hparams.vertical_trim,
        bootstrap_sample=hparams.bootstrap,
        mask_beginning_and_end=hparams.mask_beginning_and_end,
        begin_mask=hparams.begin_mask,
        end_mask=hparams.end_mask,
    )

    val_dataset = SpectrogramDatasetMultiLabel(
        val_files,
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
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=hparams.num_workers,
        collate_fn=None if hparams.batch_size == 1 else pad_batch,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=hparams.num_workers,
        collate_fn=None if hparams.batch_size == 1 else pad_batch,
    )

    in_channels = DEFAULT_SPECTROGRAM_NUM_ROWS - hparams.vertical_trim

    # TODO: refactor this to incorporate data loaders more easily.
    model_kwargs = {
        "in_channels": in_channels,
        "learning_rate": hparams.learning_rate,
        "mel": hparams.mel,
        "apply_log": hparams.log,
        "n_fft": hparams.n_fft,
        "vertical_trim": hparams.vertical_trim,
        "mask_beginning_and_end": hparams.mask_beginning_and_end,
        "begin_mask": hparams.begin_mask,
        "end_mask": hparams.end_mask,
        "train_files": train_dataset.files,
        "val_files": val_dataset.files,
    }

    if hparams.apply_attn:
        model = UNet1DAttn(**model_kwargs)
    else:
        model = UNet1D(**model_kwargs)

    save_best = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor="val_loss", filename="{epoch}-{val_loss:.5f}", mode="min", save_top_k=1
    )

    lr_monitor = pl.callbacks.lr_monitor.LearningRateMonitor()

    trainer_kwargs = {
        "gpus": hparams.gpus,
        "num_nodes": hparams.num_nodes,
        "max_epochs": hparams.epochs,
        "check_val_every_n_epoch": hparams.check_val_every_n_epoch,
        "callbacks": [save_best, lr_monitor],
        "accelerator": "ddp" if hparams.gpus else None,
        "plugins": DDPPlugin(find_unused_parameters=False) if hparams.gpus else None,
        "precision": 16 if hparams.gpus else 32,
        "default_root_dir": hparams.log_dir,
        "log_every_n_steps": 1,
        "terminate_on_nan": True,
    }

    if hparams.tune_initial_lr:
        trainer_kwargs["auto_lr_find"] = True

        trainer = pl.Trainer(**trainer_kwargs)

        trainer.tune(model, train_loader, val_loader)

    else:
        trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(model, train_loader, val_loader)

    torch.save(model.state_dict(), os.path.join(trainer.log_dir, hparams.model_name))


def main(args):
    train(args)
