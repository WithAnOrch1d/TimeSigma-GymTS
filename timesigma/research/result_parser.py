from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path

import toml

from timesigma.algorithms import (
    EASYTSAD_REVISION,
    EASYTSAD_SCHEMAS,
    get_method,
)
from timesigma.algorithms.configuration import resolve_configuration
from timesigma.benchmark.leaderboard import OFFICIAL_METRIC_IDS
from timesigma.benchmark.runner import leaderboard_items
from timesigma.profiles import build_dataset_profiles


@dataclass(frozen=True)
class _ParsedComponentResult:
    curve: str | None
    training_curves: tuple[str, ...]
    held_out_curves: tuple[str, ...]
    metrics: Mapping[str, float]
    artifact_reference: Mapping[str, object]


@dataclass(frozen=True)
class _ParsedComponentJob:
    method: str
    schema: str
    dataset: str
    configuration_name: str
    resolved_configuration: Mapping[str, object]
    evidence_reference: Mapping[str, object]
    units: tuple[_ParsedComponentResult, ...]


def parse_validated_component_results(
    job_dir: Path,
    dataset_dir: Path,
    dataset_provenance: Mapping[str, object],
    *,
    profile_dir: Path,
) -> _ParsedComponentJob:
    parsed, _ = _parse_component_results_with_profiles(
        job_dir,
        dataset_dir,
        dataset_provenance,
        profile_dir=profile_dir,
    )
    return parsed


def _parse_component_results_with_profiles(
    job_dir: Path,
    dataset_dir: Path,
    dataset_provenance: Mapping[str, object],
    *,
    profile_dir: Path,
) -> tuple[_ParsedComponentJob, tuple[Mapping[str, object], ...]]:
    job = job_dir.resolve()
    manifest = _read_mapping(job / "run.json")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise ValueError("a validated EasyTSAD result requires one capability dataset")
    dataset = datasets[0]
    if not isinstance(dataset, str):
        raise TypeError("capability dataset identity must be a string")
    if dataset.upper() == "AIOPS":
        raise ValueError("AIOPS evidence cannot enter Component Memory")

    method = _string(manifest, "method")
    schema = _string(manifest, "schema")
    configuration_name = _string(manifest, "configuration_name")
    if schema not in EASYTSAD_SCHEMAS:
        raise ValueError(f"unsupported EasyTSAD schema: {schema}")
    try:
        spec = get_method(method)
    except KeyError as exc:
        raise ValueError(f"unknown capability method: {method}") from exc
    capability = spec.capability(schema)
    if not capability.supported:
        raise ValueError(f"unsupported capability schema: {method}/{schema}")

    data_root = dataset_dir.resolve()
    if data_root.name != dataset:
        raise ValueError("explicit dataset directory does not match the run")
    source = _string(dataset_provenance, "source")
    revision = _string(dataset_provenance, "revision")
    _validate_manifest(manifest, spec.module, source, revision)
    resolved_configuration = _resolved_configuration(job, manifest, spec.module, dataset)

    membership = None
    if schema == "zero_shot":
        membership = _job_path(
            job,
            _string(manifest, "zero_shot_membership_path"),
        )
    profiles = build_dataset_profiles(
        dataset=dataset,
        dataset_dir=data_root,
        schema=schema,
        provenance={"source": source, "revision": revision},
        zero_shot_membership=membership,
        profile_dir=profile_dir,
    )

    results = _job_path(job, _string(manifest, "results_path"))
    evaluation_path = (
        results
        / "Evals"
        / method
        / schema
        / dataset
        / ("all.json" if schema == "naive" else "avg.json")
    )
    evaluations = _read_mapping(evaluation_path)
    profiles_by_curve = {profile.curve: profile for profile in profiles}
    if schema == "naive" and (
        not evaluations or not set(evaluations).issubset(profiles_by_curve)
    ):
        raise ValueError("naive evaluations do not match Dataset Profile curves")

    artifact_manifest = _mapping(manifest, "model_artifacts")
    artifact_root = _job_path(job, _string(artifact_manifest, "path"))
    artifact_root = artifact_root / method / schema / dataset
    selected_profiles = (
        tuple(profiles_by_curve[curve] for curve in sorted(evaluations))
        if schema == "naive"
        else profiles
    )
    units = []
    profile_payloads = []
    for profile in selected_profiles:
        profile_payload = profile.as_dict()
        if schema == "all_in_one" and "ordered_members" in manifest:
            ordered = manifest["ordered_members"][dataset]
            if len(ordered) != len(profile_payload["training_curves"]) or set(ordered) != set(profile_payload["training_curves"]):
                raise ValueError("frozen ALL_IN_ONE members differ from Dataset Profile")
            profile_payload["training_curves"] = list(ordered)
        evaluation = (
            _as_mapping(evaluations[profile.curve], "curve evaluation")
            if schema == "naive"
            else evaluations
        )
        metadata_path = (
            artifact_root / str(profile.curve) / "metadata.json"
            if schema == "naive"
            else artifact_root / "metadata.json"
        )
        artifact = _artifact_reference(
            job=job,
            metadata_path=metadata_path,
            method=method,
            schema=schema,
            dataset=dataset,
            curve=profile.curve,
            resolved_configuration=resolved_configuration,
            expected_artifact_type=spec.artifact_type,
            expected_selection_policy=capability.selection_policy,
            expected_selection_objective=capability.selection_objective,
            profile=profile_payload,
            expected_seed=manifest.get("seed"),
        )
        units.append(
            _ParsedComponentResult(
                curve=profile.curve,
                training_curves=tuple(profile_payload["training_curves"]),
                held_out_curves=tuple(profile_payload["held_out_curves"]),
                metrics=_official_metrics(dataset, evaluation),
                artifact_reference=artifact,
            )
        )
        profile_payloads.append(profile_payload)
    parsed = _ParsedComponentJob(
        method=method,
        schema=schema,
        dataset=dataset,
        configuration_name=configuration_name,
        resolved_configuration=deepcopy(resolved_configuration),
        evidence_reference={
            "job": str(job),
            "run": "run.json",
            "evaluation": _relative(job, evaluation_path),
            "dataset_source": source,
            "dataset_revision": revision,
            "easytsad_revision": EASYTSAD_REVISION,
            "module": spec.module,
            "fit_scope": capability.fit_scope,
            "seed": manifest.get("seed"),
        },
        units=tuple(units),
    )
    return parsed, tuple(profile_payloads)


def _validate_manifest(
    manifest: Mapping[str, object],
    module: str,
    source: str,
    revision: str,
) -> None:
    recorded = _mapping(manifest, "dataset")
    if (
        manifest.get("status") != "complete"
        or manifest.get("module") != module
        or manifest.get("easytsad_revision") != EASYTSAD_REVISION
        or recorded.get("source") != source
        or recorded.get("revision") != revision
        or recorded.get("clean") is not True
    ):
        raise ValueError("capability manifest provenance or status is mismatched")


def _resolved_configuration(
    job: Path,
    manifest: Mapping[str, object],
    module: str,
    dataset: str,
) -> dict[str, object]:
    frozen_path = _job_path(job, _string(manifest, "configuration_path"))
    frozen = toml.load(frozen_path)
    resolved = resolve_configuration(module, (dataset,), frozen)
    config = _mapping(manifest, "config")
    if (
        config.get("Data_Params") != frozen.get("Data_Params")
        or config.get("Model_Params") != frozen.get("Model_Params")
        or config.get("Resolved") != resolved
    ):
        raise ValueError("capability resolved configuration is mismatched")
    return deepcopy(resolved[dataset])


def _artifact_reference(
    *,
    job: Path,
    metadata_path: Path,
    method: str,
    schema: str,
    dataset: str,
    curve: str | None,
    resolved_configuration: Mapping[str, object],
    expected_artifact_type: str,
    expected_selection_policy: str | None,
    expected_selection_objective: str | None,
    profile: Mapping[str, object],
    expected_seed: int | None = None,
) -> dict[str, object]:
    metadata = _read_mapping(metadata_path)
    expected = {
        "method": method,
        "schema": schema,
        "dataset": dataset,
        "curve": curve,
        "configuration": resolved_configuration,
        "artifact_type": expected_artifact_type,
        "selection_policy": expected_selection_policy,
        "selection_objective": expected_selection_objective,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Model Artifact identity is mismatched: {metadata_path}")
    from timesigma.algorithms.artifacts import validate_model_artifact_metadata

    spec = get_method(method)
    defaults = resolve_configuration(spec.module, (dataset,))[dataset]["Model_Params"]
    spec.validate_parameters(resolved_configuration["Model_Params"], defaults)
    validate_model_artifact_metadata(spec, metadata, schema=schema, configuration=resolved_configuration)
    if method == "IsolationForest":
        run = _read_mapping(job / "run.json")
        if metadata["seed"] != expected_seed or metadata["dataset_revision"] != run["dataset"]["revision"]:
            raise ValueError(f"{method} Artifact seed/revision differs from its execution manifest")
        if metadata["score_support"] != _read_mapping(job / "score-support.json")[dataset]:
            raise ValueError(f"{method} Artifact differs from executed score support")
    if method in {"LSTMADalpha", "LSTMADbeta", "FITS", "DCdetector", "FCVAE"}:
        from timesigma.algorithms.artifacts import validate_score_metadata

        validate_score_metadata(get_method(method), metadata, resolved_configuration)
        if metadata.get("seed") != expected_seed:
            raise ValueError(f"{method} Artifact seed differs from its execution manifest")
        support = _read_mapping(job / "score-support.json")[dataset]
        if metadata["score_support"] != {curve: support[curve] for curve in metadata["test_curves"]}:
            raise ValueError(f"{method} Artifact differs from the executed score support")
    if metadata.get("selection_status") not in {
        "available",
        "unavailable",
        "not_applicable",
    }:
        raise ValueError(f"Model Artifact selection status is invalid: {metadata_path}")

    training_curves = profile["training_curves"]
    held_out_curves = profile["held_out_curves"]
    test_curves = held_out_curves if schema == "zero_shot" else training_curves
    if (
        list(metadata.get("training_curves", ())) != list(training_curves)
        or list(metadata.get("test_curves", ())) != list(test_curves)
    ):
        raise ValueError(f"Model Artifact members or order are mismatched: {metadata_path}")
    if schema == "zero_shot" and metadata.get("zero_shot_split") != profile.get(
        "zero_shot_split"
    ):
        raise ValueError(f"Model Artifact zero-shot split is mismatched: {metadata_path}")

    artifact_name = metadata.get("artifact_path")
    state_reference = None
    if artifact_name is not None:
        if not isinstance(artifact_name, str):
            raise TypeError("Model Artifact path must be a string or null")
        state_path = _contained_path(job, metadata_path.parent / artifact_name)
        if not state_path.is_file():
            raise FileNotFoundError(f"Model Artifact state is missing: {state_path}")
        if method == "IsolationForest":
            import joblib
            from timesigma.algorithms.isolation_forest.binding import validate_estimator

            validate_estimator(joblib.load(state_path), metadata)
        if method == "FCVAE":
            import torch
            from timesigma.algorithms.artifacts import validate_fcvae_replay
            from timesigma.algorithms.fcvae import REPLAY_KEY

            payload = torch.load(state_path, map_location="cpu", weights_only=True)
            validate_fcvae_replay(payload.get(REPLAY_KEY), len(metadata["test_curves"]))
        state_reference = _relative(job, state_path)
    return {
        "artifact_type": metadata["artifact_type"],
        "selection_status": metadata["selection_status"],
        "selection_policy": metadata["selection_policy"],
        "selection_objective": metadata["selection_objective"],
        "selection_reason": metadata.get("selection_reason"),
        "selection_values": deepcopy(metadata.get("selection_values")),
        "best_epoch": metadata.get("best_epoch"),
        "score_source": metadata.get("score_source"),
        "sample_counts": deepcopy(metadata.get("sample_counts")),
        "training_curves": deepcopy(metadata.get("training_curves")),
        "test_curves": deepcopy(metadata.get("test_curves")),
        "metadata": _relative(job, metadata_path),
        "state": state_reference,
    }


def _official_metrics(
    dataset: str,
    evaluation: Mapping[str, object],
) -> dict[str, float]:
    try:
        items = leaderboard_items(dataset)
    except KeyError as exc:
        raise ValueError(f"unsupported EasyTSAD leaderboard dataset: {dataset}") from exc
    metrics: dict[str, float] = {}
    for metric_id, (_, keys) in zip(OFFICIAL_METRIC_IDS, items, strict=True):
        value: object = evaluation
        for key in keys:
            value = _as_mapping(value, f"evaluation path {keys}").get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"official evaluation metric is not numeric: {metric_id}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"official evaluation metric is not finite: {metric_id}")
        metrics[metric_id] = numeric
    return metrics


def _read_mapping(path: Path) -> Mapping[str, object]:
    return _as_mapping(
        json.loads(path.read_text(encoding="utf-8")),
        str(path),
    )


def _job_path(job: Path, relative: str) -> Path:
    return _contained_path(job, job / relative)


def _contained_path(job: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(job):
        raise ValueError(f"capability reference escapes its job: {path}")
    return resolved


def _relative(job: Path, path: Path) -> str:
    return _contained_path(job, path).relative_to(job).as_posix()


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _as_mapping(payload.get(key), key)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


__all__ = [
    "_ParsedComponentJob",
    "_ParsedComponentResult",
    "parse_validated_component_results",
]
