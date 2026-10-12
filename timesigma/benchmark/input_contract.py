from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import toml

from timesigma.algorithms import AlgorithmSpec
from timesigma.algorithms.configuration import resolve_configuration
from .zero_shot import ZeroShotSplit, split_for_dataset


def validate_input_lengths(
    spec: AlgorithmSpec, parameters: Mapping[str, Any], schema: str,
    lengths: Mapping[str, Mapping[str, int]], training: Sequence[str],
) -> dict[str, dict[str, int]]:
    model = SimpleNamespace(**parameters)
    groups = [(curve,) for curve in training] if schema == "naive" else [tuple(training)]
    for group in groups:
        for phase in (("train",) if spec.method_name == "IsolationForest" else ("train", "valid")):
            count = sum(spec.sample_spec.count(lengths[curve][phase], model) for curve in group)
            if count == 0:
                raise ValueError(f"{spec.method_name} {phase} members {list(group)} have no complete window_size + pred_len window; no model will be fitted")
    testing = [curve for curve in lengths if curve not in training] if schema == "zero_shot" else list(lengths)
    return {curve: spec.score_support(lengths[curve]["test"], parameters) for curve in testing}


@contextmanager
def bind_input_contract(
    spec: AlgorithmSpec, schema: str, configurations: Mapping[str, Mapping[str, Any]],
    splits: Sequence[ZeroShotSplit] | None,
) -> Iterator[dict[str, dict[str, dict[str, int]]]]:
    """Admit all units after native transforms and before Method construction."""
    from EasyTSAD.TrainingSchema.BaseSchema import BaseSchema

    original = BaseSchema.load_data
    supports: dict[str, dict[str, dict[str, int]]] = {}

    def load_data(instance: Any) -> Any:
        loaded = original(instance)
        for dataset, series in loaded.items():
            parameters = configurations[dataset]["Model_Params"]
            defaults = resolve_configuration(spec.module, (dataset,))[dataset]["Model_Params"]
            spec.validate_parameters(parameters, defaults)
            training = tuple(series)
            if schema == "zero_shot":
                if splits is None:
                    raise ValueError("input admission requires frozen zero-shot membership")
                training = split_for_dataset(splits, dataset).training_curves
            lengths = {curve: {phase: len(getattr(arrays, phase)) for phase in ("train", "valid", "test")} for curve, arrays in series.items()}
            supports[dataset] = validate_input_lengths(spec, parameters, schema, lengths, training)
        return loaded

    BaseSchema.load_data = load_data
    try:
        yield supports
    finally:
        BaseSchema.load_data = original


def validate_score_support(root: Path, method: str, schema: str, supports: Mapping[str, Mapping[str, Mapping[str, int]]]) -> None:
    for dataset, curves in supports.items():
        for curve, support in curves.items():
            score = np.load(root / "Results/Scores" / method / schema / dataset / (curve + ".npy"), allow_pickle=False)
            if score.shape != (support["count"],) or not np.isfinite(score).all():
                raise ValueError(f"{method}/{dataset}/{curve} score violates declared native support")
            tail = support["native_tail_zeros"]
            if tail and not np.array_equal(score[-tail:], np.zeros(tail)):
                raise ValueError(f"{method}/{dataset}/{curve} lost its native tail zeros")
    (root / "score-support.json").write_text(json.dumps(supports, indent=2, sort_keys=True) + "\n")


def validate_research_input_lengths(pipeline: Any, task: Any, dataset_dir: Path) -> None:
    if pipeline.method not in ("LSTMADalpha", "LSTMADbeta", "FITS", "DCdetector", "FCVAE", "IsolationForest"):
        return
    if task.dataset.casefold() == "aiops":
        raise ValueError("research length preview cannot inspect AIOPS test-side metadata")
    from importlib.resources import files
    from timesigma.algorithms import get_method

    settings = toml.loads((files("EasyTSAD.Controller") / "GlobalCfg.toml").read_text())["DatasetSetting"]
    lengths = {}
    for curve in (*task.unit.training_curves, *task.unit.held_out_curves):
        train = len(np.load(dataset_dir / curve / "train.npy", mmap_mode="r", allow_pickle=False))
        proportion = settings["train_proportion"]
        if 0 < proportion < 1:
            train = int(train * proportion) or train  # native [-0:] keeps the whole train
        valid = train
        proportion = settings["valid_proportion"]
        if 0 < proportion < 1:
            size = int(train * proportion)
            train, valid = (train - size, size) if size else (0, train)
        lengths[curve] = {"train": train, "valid": valid, "test": len(np.load(dataset_dir / curve / "test.npy", mmap_mode="r", allow_pickle=False))}
    validate_input_lengths(get_method(pipeline.method), pipeline.resolved_configuration["Model_Params"], task.schema, lengths, task.unit.training_curves)
