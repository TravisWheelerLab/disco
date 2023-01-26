from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data.dataset import Dataset


class DataModule(Dataset, ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def collate_fn(self):
        """
        Return None for default collate function.
        Otherwise, return a function that operates on batch.

        :return:
        :rtype:
        """
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        pass
