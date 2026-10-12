from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

import numpy as np

from timesigma.benchmark.zero_shot import ZeroShotSplit, split_for_dataset


def transform_series(
    series: Mapping[str, Any],
    operators: Sequence[Mapping[str, Any]],
    training_curves: Sequence[str],
    *,
    shared: bool,
) -> list[dict[str, Any]]:
    states = []
    for operator in operators:
        name, parameters = operator["component_id"], operator["parameters"]
        state: dict[str, Any] = {"component_id": name, "parameters": dict(parameters)}
        if name == "robust_scaler":
            groups = (
                {"shared": tuple(training_curves)}
                if shared else {curve: (curve,) for curve in training_curves}
            )
            fitted = {}
            for key, curves in groups.items():
                values = np.concatenate([series[curve].train for curve in curves])
                if not values.size or not np.isfinite(values).all():
                    raise ValueError("scaler fit requires finite non-empty training values")
                q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75))
                scale, source = float(q75 - q25), "iqr"
                if scale == 0:
                    scale, source = float(np.std(values)), "std"
                if scale == 0:
                    scale, source = 1.0, "unit"
                fitted[key] = {
                    "median": float(median),
                    "scale": scale,
                    "scale_source": source,
                    "training_curves": list(curves),
                    "sample_counts": {curve: len(series[curve].train) for curve in curves},
                }
            for curve, arrays in series.items():
                fit = fitted["shared" if shared else curve]
                for split in ("train", "valid", "test"):
                    values = np.asarray(getattr(arrays, split))
                    setattr(arrays, split, (values - fit["median"]) / fit["scale"])
            state["fitted"] = fitted
        elif name == "moving_average":
            window = parameters["window"]
            for arrays in series.values():
                for split in ("train", "valid", "test"):
                    values = np.asarray(getattr(arrays, split))
                    if not values.size:
                        continue
                    left = window // 2
                    padded = np.pad(values, (left, window - 1 - left), mode="edge")
                    smoothed = np.convolve(
                        padded, np.full(window, 1.0 / window), mode="valid",
                    )
                    setattr(arrays, split, smoothed)
            state["temporal_policy"] = "split_local_centered_edge"
        else:
            raise ValueError(f"unsupported Pipeline input component: {name}")
        states.append(state)
    return states


@contextmanager
def bind_pipeline_inputs(
    *,
    configuration: Mapping[str, Any],
    schema: str,
    directory: Path,
    splits: Sequence[ZeroShotSplit] | None,
) -> Iterator[None]:
    from EasyTSAD.TrainingSchema.BaseSchema import BaseSchema

    original = BaseSchema.load_data

    def load_data(instance: Any) -> Any:
        loaded = original(instance)
        if (
            instance.schema != configuration["schema"]
            or set(loaded) != {configuration["dataset"]}
        ):
            raise ValueError("Pipeline input scope does not match its frozen configuration")
        states = {}
        for dataset, series in loaded.items():
            expected = set(configuration["training_curves"]) | set(configuration["held_out_curves"])
            if set(series) != expected:
                raise ValueError("Pipeline input members do not match the frozen PlanningUnit")
            if schema == "all_in_one" and list(series) != list(configuration["training_curves"]):
                raise ValueError("Pipeline input member order differs from the frozen PlanningUnit")
            training = tuple(series)
            if schema == "zero_shot":
                if splits is None:
                    raise ValueError("Pipeline preprocessing requires frozen zero-shot members")
                training = split_for_dataset(splits, dataset).training_curves
            states[dataset] = {
                "split": {
                    key: instance.dc[key]
                    for key in ("train_proportion", "valid_proportion")
                },
                "components": transform_series(
                    series, json.loads(configuration["operators_json"]),
                    training, shared=schema != "naive",
                ),
            }
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "preprocessing-state.json").write_text(
            json.dumps(states, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return loaded

    BaseSchema.load_data = load_data
    try:
        yield
    finally:
        BaseSchema.load_data = original
