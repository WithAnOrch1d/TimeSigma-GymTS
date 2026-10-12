from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import toml

from timesigma.algorithms import get_method
from timesigma.algorithms.artifacts import (
    resolve_selection_status,
    validate_score_lineage,
)
from timesigma.algorithms.configuration import resolve_configuration

from ..runner import SUMMARY_ITEMS
from ..zero_shot import (
    MEMBERSHIP_FILENAME,
    ZeroShotSplit,
    load_zero_shot_membership,
    split_for_dataset,
    validate_zero_shot_scores,
)


def validate_capability_job(
    directory: Path,
    dataset_root: Path,
    expected_run: Mapping[str, object],
    expected_membership: Path | None,
) -> None:
    run_path = directory / "run.json"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    identity = {key: value for key, value in expected_run.items() if key != "dataset"}
    actual_dataset = payload.get("dataset")
    expected_dataset = expected_run.get("dataset")
    dataset_matches = (
        isinstance(actual_dataset, Mapping)
        and isinstance(expected_dataset, Mapping)
        and bool(actual_dataset.get("root"))
        and all(
            actual_dataset.get(key) == expected_dataset.get(key)
            for key in ("source", "revision", "clean")
        )
    )
    if (
        any(payload.get(key) != value for key, value in identity.items())
        or not dataset_matches
    ):
        raise ValueError(
            f"capability run identity is incomplete or mismatched: {run_path}"
        )
    method = str(expected_run["method"])
    schema = str(expected_run["schema"])
    dataset = str(expected_run["datasets"][0])
    config = payload.get("config")
    frozen_path = directory / "configuration.toml"
    if (
        not payload.get("runtime_python")
        or not isinstance(config, dict)
        or not {"Data_Params", "Model_Params", "Resolved"} <= set(config)
        or not frozen_path.is_file()
    ):
        raise ValueError(f"capability run provenance is incomplete: {run_path}")
    frozen = toml.load(frozen_path)
    resolved = resolve_configuration(str(expected_run["module"]), (dataset,), frozen)
    if (
        config["Data_Params"] != frozen.get("Data_Params")
        or config["Model_Params"] != frozen.get("Model_Params")
        or config["Resolved"] != resolved
    ):
        raise ValueError(f"capability configuration is mismatched: {frozen_path}")

    results = directory / "Results"
    score_dir = results / "Scores" / method / schema / dataset
    score_paths = tuple(sorted(score_dir.glob("*.npy")))
    if schema == "zero_shot":
        split = split_for_dataset(
            load_zero_shot_membership(results / MEMBERSHIP_FILENAME), dataset
        )
        if expected_membership is not None:
            expected = split_for_dataset(
                load_zero_shot_membership(expected_membership), dataset
            )
            if split != expected:
                raise ValueError(f"zero-shot membership differs for {dataset}")
        validate_zero_shot_scores(results, (method,), (split,))
        expected_curves = set(split.held_out_curves)
    else:
        expected_curves = {
            path.name
            for path in (dataset_root / "UTS" / dataset).iterdir()
            if path.is_dir()
        }
    if {path.stem for path in score_paths} != expected_curves:
        raise RuntimeError(
            f"capability score members are incomplete: {method}/{schema}/{dataset}"
        )
    for path in score_paths:
        score = np.load(path, mmap_mode="r", allow_pickle=False)
        test = np.load(
            dataset_root / "UTS" / dataset / path.stem / "test.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if score.ndim != 1 or score.size == 0 or score.size > test.shape[0]:
            raise RuntimeError(f"invalid capability score shape: {path}")
        for start in range(0, score.size, 1_000_000):
            if not np.isfinite(score[start : start + 1_000_000]).all():
                raise RuntimeError(f"non-finite capability score: {path}")

    model_artifacts = expected_run.get("model_artifacts")
    if model_artifacts is not None:
        if not isinstance(model_artifacts, Mapping):
            raise TypeError("capability Model Artifact identity must be a mapping")
        _validate_model_artifacts(
            directory=directory,
            method=method,
            schema=schema,
            dataset=dataset,
            score_curves=tuple(path.stem for path in score_paths),
            all_curves=tuple(
                path.name
                for path in (dataset_root / "UTS" / dataset).iterdir()
                if path.is_dir()
            ),
            model_artifacts=model_artifacts,
            split=(split if schema == "zero_shot" else None),
            configuration=config["Resolved"][dataset],
        )

    eval_dir = results / "Evals" / method / schema / dataset
    runtime_path = results / "RunTime" / method / schema / dataset / "time.json"
    for path in (eval_dir / "all.json", eval_dir / "avg.json", runtime_path):
        if not path.is_file():
            raise FileNotFoundError(f"capability artifact is missing: {path}")
    _validate_summary(
        results / "Summary" / "CSVs" / f"{schema}.csv", method, dataset
    )


def _validate_model_artifacts(
    *,
    directory: Path,
    method: str,
    schema: str,
    dataset: str,
    score_curves: tuple[str, ...],
    all_curves: tuple[str, ...],
    model_artifacts: Mapping[str, object],
    split: ZeroShotSplit | None,
    configuration: Mapping[str, object],
) -> None:
    spec = get_method(method)
    capability = spec.capability(schema)
    root = directory / str(model_artifacts["path"])
    method_root = root / method / schema / dataset
    artifact_dirs = (
        tuple(method_root / curve for curve in score_curves)
        if schema == "naive"
        else (method_root,)
    )
    for artifact_dir in artifact_dirs:
        metadata_path = artifact_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            raise ValueError(
                f"capability Model Artifact metadata is invalid: {metadata_path}"
            )
        expected = {
            "method": method,
            "schema": schema,
            "dataset": dataset,
            "artifact_type": spec.artifact_type,
            "selection_policy": capability.selection_policy,
            "selection_objective": capability.selection_objective,
            "configuration": configuration,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"capability Model Artifact identity is invalid: {metadata_path}"
            )
        if schema == "naive" and metadata.get("curve") != artifact_dir.name:
            raise ValueError(
                f"capability Model Artifact curve is invalid: {metadata_path}"
            )

        expected_training = [artifact_dir.name] if schema == "naive" else list(all_curves)
        expected_testing = list(expected_training)
        if split is not None:
            expected_training = list(split.training_curves)
            expected_testing = list(split.held_out_curves)
            if metadata.get("zero_shot_split") != split.as_dict():
                raise ValueError(
                    f"capability zero-shot artifact split is invalid: {metadata_path}"
                )
        if set(metadata.get("training_curves", ())) != set(expected_training):
            raise ValueError(
                f"capability artifact training curves are invalid: {metadata_path}"
            )
        if set(metadata.get("test_curves", ())) != set(expected_testing):
            raise ValueError(
                f"capability artifact test curves are invalid: {metadata_path}"
            )

        counts = metadata.get("sample_counts")
        required_counts = {
            "train_points",
            "valid_points",
            "test_points",
            "train_samples",
            "valid_samples",
            "test_samples",
            "fit_samples",
        }
        if (
            not isinstance(counts, Mapping)
            or set(counts) != required_counts
            or not all(isinstance(value, int) and value >= 0 for value in counts.values())
            or counts["test_samples"] == 0
        ):
            raise ValueError(
                f"capability Model Artifact sample counts are invalid: {metadata_path}"
            )
        sample_counts = {key: int(counts[key]) for key in required_counts}
        selection_status, selection_reason = resolve_selection_status(
            capability,
            sample_counts,
        )
        if (
            metadata.get("selection_status") != selection_status
            or metadata.get("selection_reason") != selection_reason
        ):
            raise ValueError(
                f"capability Model Artifact selection is invalid: {metadata_path}"
            )
        validate_score_lineage(
            selection_status,
            spec.artifact_type,
            metadata.get("score_source"),
        )
        _validate_selection(
            metadata,
            artifact_dir,
            selection_status,
            spec.artifact_type,
        )


def _validate_selection(
    metadata: Mapping[str, object],
    artifact_dir: Path,
    status: str,
    artifact_type: str,
) -> None:
    artifact_path = metadata.get("artifact_path")
    selection_values = metadata.get("selection_values")
    best_epoch = metadata.get("best_epoch")
    if status == "available":
        values = np.asarray(selection_values, dtype=float)
        if (
            artifact_type != "torch_state_dict"
            or artifact_path != "best.pt"
            or not (artifact_dir / "best.pt").is_file()
            or values.ndim != 1
            or values.size == 0
            or not np.isfinite(values).all()
            or not isinstance(best_epoch, int)
            or best_epoch < 1
        ):
            raise ValueError(f"available Model Artifact is invalid: {artifact_dir}")
        return
    if status == "unavailable":
        if (
            artifact_type != "torch_state_dict"
            or artifact_path is not None
            or selection_values is not None
            or best_epoch is not None
        ):
            raise ValueError(f"unavailable Model Artifact is invalid: {artifact_dir}")
        return
    expected_path = (
        "estimator.pkl" if artifact_type == "fitted_estimator" else None
    )
    if (
        status != "not_applicable"
        or artifact_path != expected_path
        or (expected_path is not None and not (artifact_dir / expected_path).is_file())
        or selection_values is not None
        or best_epoch is not None
    ):
        raise ValueError(f"not-applicable Model Artifact is invalid: {artifact_dir}")


def _validate_summary(path: Path, method: str, dataset: str) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3:
        raise ValueError(f"capability Summary is incomplete: {path}")
    indices = [index for index, value in enumerate(rows[0]) if value == dataset]
    columns = [item[0] for item in SUMMARY_ITEMS]
    if len(indices) != 16 or [rows[1][index] for index in indices] != columns:
        raise ValueError(
            f"capability Summary does not contain 16 official columns: {path}"
        )
    matches = [row for row in rows[2:] if row and row[0].strip() == method]
    if len(matches) != 1 or not all(
        np.isfinite(float(matches[0][index])) for index in indices
    ):
        raise ValueError(f"capability Summary row is incomplete: {method}/{dataset}")


__all__ = ["validate_capability_job"]
