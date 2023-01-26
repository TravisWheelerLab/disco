from __future__ import annotations

import pytorch_lightning as pl

from disco_sound import datasets, models
from disco_sound.util import pluginloader


def load_models() -> dict[str, type[pl.LightningModule]]:
    return {
        m.__name__: m
        for m in pluginloader.load_plugin_classes(models, pl.LightningModule)
    }


def load_datasets() -> dict[str, type[datasets.DataModule]]:
    return {
        m.__name__: m
        for m in pluginloader.load_plugin_classes(datasets, datasets.DataModule)
    }


def _get_dataset(name: str) -> type[datasets.DataModule]:
    dataset_dict = load_datasets()

    if name not in dataset_dict:
        raise ValueError(f"Dataset {name} not found.")

    return dataset_dict[name]


def _get_model(name: str) -> type[pl.LightningModule]:
    model_dict = load_models()

    if name not in model_dict:
        raise ValueError(f"Model {name} not found.")

    return model_dict[name]


def load_dataset_class(dataset_name):
    dataset_cls = _get_dataset(dataset_name)
    return dataset_cls


def load_model_class(model_name):
    model_cls = _get_model(model_name)
    return model_cls
