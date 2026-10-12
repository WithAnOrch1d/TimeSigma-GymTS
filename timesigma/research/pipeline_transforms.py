from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


def apply_preprocessing_components(
    *,
    train: np.ndarray,
    test: np.ndarray,
    components: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    transformed_train = _series(train, "train")
    transformed_test = _series(test, "test")
    for component in components:
        component_id = _component_id(component)
        parameters = _parameters(component)
        if component_id == "robust_scaler":
            transformed_train, transformed_test = _robust_scale(
                transformed_train,
                transformed_test,
            )
        elif component_id == "moving_average":
            window = _positive_int(parameters.get("window", 5), "moving_average.window")
            transformed_train = _moving_average(transformed_train, window)
            transformed_test = _moving_average(transformed_test, window)
        elif component_id in {"native_preprocess", "native_diff_order"}:
            continue
        else:
            raise ValueError(f"unsupported preprocessing component: {component_id}")
    return transformed_train, transformed_test


def materialize_dataset_view(
    *,
    dataset: str,
    dataset_dir: Path,
    output_root: Path,
    curve_names: Sequence[str],
    preprocessing: Sequence[Mapping[str, Any]],
) -> Path:
    if not dataset.strip():
        raise ValueError("dataset is required")
    if not curve_names:
        raise ValueError("materialized dataset view requires at least one curve")
    source_root = dataset_dir.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"dataset directory is missing: {source_root}")
    destination_root = (output_root / "UTS" / dataset).resolve()
    if destination_root.exists() and any(destination_root.iterdir()):
        raise FileExistsError(f"dataset view already exists: {destination_root}")
    destination_root.mkdir(parents=True, exist_ok=True)

    for curve in tuple(curve_names):
        if not isinstance(curve, str) or not curve.strip():
            raise ValueError("curve names must be non-empty strings")
        source = source_root / curve
        if not source.is_dir():
            raise FileNotFoundError(f"source curve is missing: {source}")
        destination = destination_root / curve
        destination.mkdir()
        train, test = apply_preprocessing_components(
            train=np.load(source / "train.npy", allow_pickle=False),
            test=np.load(source / "test.npy", allow_pickle=False),
            components=preprocessing,
        )
        np.save(destination / "train.npy", train, allow_pickle=False)
        np.save(destination / "test.npy", test, allow_pickle=False)
        for name in ("train_label.npy", "test_label.npy", "info.json"):
            path = source / name
            if path.is_file():
                shutil.copy2(path, destination / name)

    _write_json(
        output_root / "dataset-view.json",
        {
            "dataset": dataset,
            "curve_names": list(curve_names),
            "preprocessing": deepcopy(list(preprocessing)),
            "output_dataset_root": str(output_root),
        },
    )
    return output_root.resolve()


def apply_score_postprocessing(
    scores: np.ndarray,
    components: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    transformed = _series(scores, "scores")
    for component in components:
        component_id = _component_id(component)
        parameters = _parameters(component)
        if component_id == "score_moving_average":
            window = _positive_int(parameters.get("window", 5), "score_moving_average.window")
            transformed = _moving_average(transformed, window)
        else:
            raise ValueError(f"unsupported score postprocessing component: {component_id}")
    return transformed


def fuse_scores_mean(scores: Sequence[np.ndarray]) -> np.ndarray:
    if not scores:
        raise ValueError("mean score fusion requires at least one score array")
    arrays = tuple(_series(score, "score") for score in scores)
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("mean score fusion requires equal score shapes")
    return np.mean(np.stack(arrays, axis=0), axis=0)


def _robust_scale(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = float(np.median(train))
    q25, q75 = (float(value) for value in np.quantile(train, (0.25, 0.75)))
    scale = q75 - q25
    if scale == 0.0:
        scale = float(np.std(train))
    if scale == 0.0:
        scale = 1.0
    return (train - median) / scale, (test - median) / scale


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window == 1:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def _series(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array


def _component_id(component: Mapping[str, Any]) -> str:
    return _text(component.get("component_id"), "component_id")


def _parameters(component: Mapping[str, Any]) -> Mapping[str, Any]:
    value = component.get("parameters", {})
    if not isinstance(value, Mapping):
        raise TypeError("component parameters must be a mapping")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 1:
        raise ValueError(f"{label} must be positive")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "apply_preprocessing_components",
    "apply_score_postprocessing",
    "fuse_scores_mean",
    "materialize_dataset_view",
]
