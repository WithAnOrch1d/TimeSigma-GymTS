"""Freeze and verify EasyTSAD zero-shot curve membership."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from importlib import import_module
import json
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

import numpy as np

from EasyTSAD.Controller import TSADController
from EasyTSAD.TrainingSchema.ZeroShot import dict_split


MEMBERSHIP_FILENAME = "zero_shot_membership.json"


@dataclass(frozen=True)
class ZeroShotSplit:
    dataset: str
    proportion: float
    random_seed: int
    enumerated_curves: tuple[str, ...]
    training_curves: tuple[str, ...]
    held_out_curves: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "proportion": self.proportion,
            "random_seed": self.random_seed,
            "enumerated_curves": list(self.enumerated_curves),
            "training_curves": list(self.training_curves),
            "held_out_curves": list(self.held_out_curves),
        }


def calculate_zero_shot_splits(
    controller: TSADController,
    datasets: Sequence[str],
) -> tuple[ZeroShotSplit, ...]:
    state = np.random.get_state()
    try:
        splits = []
        for dataset in datasets:
            curves = tuple(
                controller.pm.get_dataset_curves(controller.dc["dataset_type"], dataset)
            )
            transfer = controller.dc["Transfer"].get(
                dataset, controller.dc["Transfer"]["Default"]
            )
            training, held_out = dict_split(
                controller.logger,
                dict.fromkeys(curves),
                transfer["proportion"],
                transfer["random_seed"],
            )
            splits.append(
                ZeroShotSplit(
                    dataset=dataset,
                    proportion=float(transfer["proportion"]),
                    random_seed=int(transfer["random_seed"]),
                    enumerated_curves=curves,
                    training_curves=tuple(training),
                    held_out_curves=tuple(held_out),
                )
            )
        return tuple(splits)
    finally:
        np.random.set_state(state)


def freeze_zero_shot_membership(
    controller: TSADController,
    datasets: Sequence[str],
    results_dir: Path,
    expected_path: Path | None = None,
) -> tuple[ZeroShotSplit, ...]:
    if expected_path is None:
        splits = calculate_zero_shot_splits(controller, datasets)
    else:
        expected = {
            split.dataset: split
            for split in load_zero_shot_membership(expected_path)
        }
        try:
            splits = tuple(expected[dataset] for dataset in datasets)
        except KeyError:
            raise ValueError("zero-shot membership does not match the frozen input") from None
        for split in splits:
            actual = controller.pm.get_dataset_curves(controller.dc["dataset_type"], split.dataset)
            transfer = controller.dc["Transfer"].get(split.dataset, controller.dc["Transfer"]["Default"])
            if set(actual) != set(split.enumerated_curves) or transfer["proportion"] != split.proportion or transfer["random_seed"] != split.random_seed:
                raise ValueError("zero-shot membership does not match the frozen input")

    path = results_dir / MEMBERSHIP_FILENAME
    path.write_text(
        json.dumps(
            {
                "schema": "zero_shot",
                "dataset_type": controller.dc["dataset_type"],
                "datasets": [split.as_dict() for split in splits],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return splits


@contextmanager
def bind_zero_shot_membership(splits: Sequence[ZeroShotSplit] | None):
    """Replay frozen members through the native trainer; never reselect members."""
    if splits is None:
        yield
        return
    module = import_module("EasyTSAD.TrainingSchema.ZeroShot")
    original = module.ZeroShot.do_exp
    by_dataset = {split.dataset: split for split in splits}

    def do_exp(instance, tsDatas, hparams=None):
        for dataset, series in tsDatas.items():
            split = by_dataset[dataset]

            def frozen_split(logger, src, proportion, seed=1):
                if set(src) != set(split.enumerated_curves) or proportion != split.proportion or seed != split.random_seed:
                    raise ValueError("execution differs from frozen zero-shot membership")
                # Native dict_split consumes these RNG draws before training. Keep
                # that RNG state, but membership comes only from the frozen file.
                np.random.seed(seed)
                np.random.shuffle(np.arange(len(src)))
                return ({name: src[name] for name in split.training_curves},
                        {name: src[name] for name in split.held_out_curves})

            with patch.object(module, "dict_split", frozen_split):
                original(instance, {dataset: series}, hparams=hparams)

    with patch.object(module.ZeroShot, "do_exp", do_exp):
        yield


def load_zero_shot_membership(path: Path) -> tuple[ZeroShotSplit, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "zero_shot" or payload.get("dataset_type") != "UTS":
        raise ValueError(f"invalid zero-shot membership file: {path}")
    try:
        splits = tuple(
            ZeroShotSplit(
                dataset=row["dataset"],
                proportion=float(row["proportion"]),
                random_seed=int(row["random_seed"]),
                enumerated_curves=_curve_names(row["enumerated_curves"]),
                training_curves=_curve_names(row["training_curves"]),
                held_out_curves=_curve_names(row["held_out_curves"]),
            )
            for row in payload["datasets"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid zero-shot membership file: {path}") from exc

    for split in splits:
        training = set(split.training_curves)
        held_out = set(split.held_out_curves)
        if training & held_out or training | held_out != set(split.enumerated_curves):
            raise ValueError(f"invalid zero-shot partition for {split.dataset}: {path}")
    if len({split.dataset for split in splits}) != len(splits):
        raise ValueError(f"duplicate dataset in zero-shot membership file: {path}")
    return splits


def split_for_dataset(
    splits: Sequence[ZeroShotSplit], dataset: str
) -> ZeroShotSplit:
    matches = [split for split in splits if split.dataset == dataset]
    if len(matches) != 1:
        raise ValueError(f"zero-shot membership has no unique entry for {dataset}")
    return matches[0]


def validate_zero_shot_scores(
    results_dir: Path,
    methods: Sequence[str],
    splits: Sequence[ZeroShotSplit],
) -> None:
    for method in methods:
        for split in splits:
            score_dir = results_dir / "Scores" / method / "zero_shot" / split.dataset
            actual = {path.stem for path in score_dir.glob("*.npy")}
            if actual != set(split.held_out_curves):
                raise RuntimeError(
                    f"zero-shot score members do not match the frozen split: "
                    f"{method}/{split.dataset}"
                )


def _curve_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError("curve names must be a list of strings")
    if len(set(value)) != len(value):
        raise ValueError("curve names must be unique")
    return tuple(value)


__all__ = [
    "MEMBERSHIP_FILENAME",
    "ZeroShotSplit",
    "calculate_zero_shot_splits",
    "freeze_zero_shot_membership",
    "load_zero_shot_membership",
    "split_for_dataset",
    "validate_zero_shot_scores",
]
