from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np


@contextmanager
def label_free_loading() -> Iterator[None]:
    """Keep native splits/transforms while withholding label and info files."""
    from EasyTSAD.Controller import PathManager
    from EasyTSAD.DataFactory import TSData

    original = TSData.__dict__["buildfrom"]

    @classmethod
    def buildfrom(cls, types, dataset, data_name, train_proportion=1, valid_proportion=0):
        manager = PathManager.get_instance()
        train = np.load(manager.get_dataset_train_set(types, dataset, data_name), allow_pickle=False)
        test = np.load(manager.get_dataset_test_set(types, dataset, data_name), allow_pickle=False)
        # Match the pinned EasyTSAD buildfrom split, including integer rounding.
        if 0 < train_proportion < 1:
            train = train[-int(len(train) * train_proportion):]
        valid = train
        if 0 < valid_proportion < 1:
            split = int(len(train) * valid_proportion)
            train, valid = train[:-split], train[-split:]
        series = cls(train, valid, test, None, None, None, None)
        for name in ("train_label", "valid_label", "test_label", "info"):
            delattr(series, name)
        return series

    TSData.buildfrom = buildfrom
    try:
        yield
    finally:
        TSData.buildfrom = original
