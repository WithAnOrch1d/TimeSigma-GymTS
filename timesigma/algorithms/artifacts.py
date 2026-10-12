from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any, Literal

from .catalog import (
    BEST_UNLABELED_VALIDATION,
    AlgorithmSpec,
    ArtifactType,
    SchemaCapability,
    SelectionStatus,
)
from .configuration import validate_parameter_type

ScoreSource = Literal[
    "selected_artifact",
    "upstream_final_state",
    "fitted_estimator",
    "recipe_execution",
]


def resolve_selection_status(
    capability: SchemaCapability,
    sample_counts: Mapping[str, int],
) -> tuple[SelectionStatus, str | None]:
    if capability.selection_policy is None:
        return "not_applicable", None
    if capability.selection_objective is None:
        raise ValueError("model-selection policy requires a declared objective")
    if capability.selection_policy == BEST_UNLABELED_VALIDATION:
        if sample_counts["valid_samples"] == 0:
            return "unavailable", "no_validation_samples"
        return "available", None
    raise ValueError(
        f"unsupported model-selection policy: {capability.selection_policy}"
    )


def score_source_for_execution(
    status: SelectionStatus,
    artifact_type: ArtifactType,
) -> ScoreSource:
    if status == "available" and artifact_type in ("torch_state_dict", "tensorflow_checkpoint"):
        return "selected_artifact"
    if status in ("unavailable", "not_applicable") and artifact_type == "torch_state_dict":
        return "upstream_final_state"
    if status == "not_applicable" and artifact_type == "fitted_estimator":
        return "fitted_estimator"
    if status == "not_applicable" and artifact_type == "replay_recipe":
        return "recipe_execution"
    raise ValueError(
        f"selection status {status} is incompatible with artifact type {artifact_type}"
    )


def validate_score_lineage(
    status: SelectionStatus,
    artifact_type: ArtifactType,
    score_source: object,
) -> None:
    allowed: tuple[ScoreSource, ...]
    if status == "available" and artifact_type == "tensorflow_checkpoint":
        allowed = ("selected_artifact",)
    elif status == "available" and artifact_type == "torch_state_dict":
        allowed = ("selected_artifact", "upstream_final_state")
    elif status in ("unavailable", "not_applicable") and artifact_type == "torch_state_dict":
        allowed = ("upstream_final_state",)
    elif status == "not_applicable" and artifact_type == "fitted_estimator":
        allowed = ("fitted_estimator",)
    elif status == "not_applicable" and artifact_type == "replay_recipe":
        allowed = ("recipe_execution",)
    else:
        raise ValueError(
            f"selection status {status} is incompatible with artifact type {artifact_type}"
        )
    if score_source not in allowed:
        raise ValueError("score source is inconsistent with the artifact run")


def save_fitted_estimator(
    method: Any,
    spec: AlgorithmSpec,
    destination: Path,
) -> None:
    if spec.artifact_type != "fitted_estimator" or spec.artifact_attribute is None:
        raise TypeError(f"{spec.method_name} has no fitted-estimator artifact")

    import joblib

    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(getattr(method, spec.artifact_attribute), destination)
    setattr(method, spec.artifact_attribute, joblib.load(destination))


def load_model_artifact(
    method: Any,
    spec: AlgorithmSpec,
    metadata_path: Path,
    *,
    schema: str,
    configuration: Mapping[str, object],
) -> bool:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_model_artifact_metadata(spec, metadata, schema=schema, configuration=configuration)
    artifact_path = metadata.get("artifact_path")
    if artifact_path is None:
        return False
    source = metadata_path.parent / str(artifact_path)
    if spec.artifact_type == "tensorflow_checkpoint":
        from third_party.adapters.omnianomaly.binding import load_checkpoint_bundle

        load_checkpoint_bundle(source, metadata, method)
        return True
    if spec.artifact_type == "torch_state_dict":
        from .checkpoint import load_checkpoint

        load_checkpoint(method, source)
        if spec.method_name == "FCVAE":
            validate_fcvae_replay(method._fcvae_replay, len(metadata["test_curves"]))
        return True
    if spec.artifact_type == "fitted_estimator" and spec.artifact_attribute is not None:
        import joblib

        estimator = joblib.load(source)
        if spec.method_name == "IsolationForest":
            from .isolation_forest.binding import validate_estimator

            validate_estimator(estimator, metadata)
        setattr(method, spec.artifact_attribute, estimator)
        return True
    raise ValueError(f"unexpected artifact file for {spec.method_name}")


def validate_model_artifact_metadata(
    spec: AlgorithmSpec,
    metadata: Mapping[str, Any],
    *,
    schema: str,
    configuration: Mapping[str, object],
) -> None:
    """The strict loader's scientific contract, also used before Evidence reuse."""
    if metadata.get("method") != spec.method_name:
        raise ValueError(f"model artifact method does not match {spec.method_name}")
    if metadata.get("artifact_type") != spec.artifact_type:
        raise ValueError(f"model artifact type does not match {spec.method_name}")
    if metadata.get("schema") != schema:
        raise ValueError(f"model artifact schema does not match {schema}")
    if metadata.get("configuration") != dict(configuration):
        raise ValueError(
            "model artifact configuration does not match the frozen pipeline"
        )
    # Python equality alone considers False == 0, including inside dictionaries.
    for section in ("Data_Params", "Model_Params"):
        for name, value in configuration[section].items():
            validate_parameter_type(value, metadata["configuration"][section][name], name)
    validate_score_metadata(spec, metadata, configuration)
    if spec.method_name == "OmniAnomaly":
        from third_party.adapters.omnianomaly.binding import validate_metadata

        validate_metadata(spec, metadata, configuration)
    if spec.method_name == "IsolationForest":
        from .isolation_forest.binding import validate_metadata

        validate_metadata(spec, metadata, configuration)

    capability = spec.capability(schema)
    if not capability.supported:
        raise ValueError(f"model artifact schema is not supported by {spec.method_name}")
    if metadata.get("selection_policy") != capability.selection_policy:
        raise ValueError("model artifact selection policy does not match the catalog")
    if metadata.get("selection_objective") != capability.selection_objective:
        raise ValueError("model artifact selection objective does not match the catalog")
    sample_counts = metadata.get("sample_counts")
    if not isinstance(sample_counts, Mapping):
        raise ValueError("model artifact sample counts are missing")
    valid_samples = sample_counts.get("valid_samples")
    if type(valid_samples) is not int or valid_samples < 0:
        raise ValueError("model artifact validation sample count is invalid")
    status, reason = resolve_selection_status(
        capability,
        {"valid_samples": valid_samples},
    )
    if (
        metadata.get("selection_status") != status
        or metadata.get("selection_reason") != reason
    ):
        raise ValueError("model artifact selection facts do not match the catalog")

    artifact_path = metadata.get("artifact_path")
    expected_path = {
        "available": "tensorflow/bundle.json" if spec.artifact_type == "tensorflow_checkpoint" else "best.pt",
        "unavailable": None,
        "not_applicable": (
            "final.pt" if spec.artifact_type == "torch_state_dict" else
            "estimator.pkl"
            if spec.artifact_type == "fitted_estimator"
            else None
        ),
    }[status]
    if artifact_path != expected_path:
        raise ValueError("model artifact path does not match its selection facts")
    validate_score_lineage(status, spec.artifact_type, metadata.get("score_source"))


def validate_fcvae_replay(replay: object, member_count: int) -> None:
    import torch

    if not isinstance(replay, Mapping) or any(not isinstance(states, list) for states in replay.values()) or sum(len(states) for states in replay.values()) != member_count:
        raise ValueError("FCVAE Artifact lacks complete per-member inference RNG")
    for states in replay.values():
        for state in states:
            if not isinstance(state, Mapping) or set(state) not in ({"cpu"}, {"cpu", "cuda"}):
                raise ValueError("FCVAE Artifact inference RNG libraries are invalid")
            if any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or not value.numel() for value in state.values()):
                raise ValueError("FCVAE Artifact inference RNG must contain byte states")


def validate_score_metadata(spec: AlgorithmSpec, metadata: Mapping[str, object], configuration: Mapping[str, object]) -> None:
    if spec.method_name not in ("LSTMADalpha", "LSTMADbeta", "FITS", "DCdetector", "FCVAE"):
        return
    if metadata.get("source_identity") != "timesigma-selected":
        raise ValueError(f"{spec.method_name} selected Artifact requires timesigma-selected source identity")
    if type(metadata.get("seed")) is not int or metadata["seed"] < 0:
        raise ValueError(f"{spec.method_name} Artifact requires its frozen seed")
    support = metadata.get("score_support")
    if not isinstance(support, Mapping) or set(support) != set(metadata["test_curves"]):
        raise ValueError(f"{spec.method_name} Artifact score support does not match test members")
    for curve, item in support.items():
        if not isinstance(item, Mapping) or type(item.get("stop")) is not int or dict(item) != spec.score_support(item["stop"], configuration["Model_Params"]):
            raise ValueError(f"{spec.method_name} Artifact score support is invalid for {curve}")
    if (sum(item["stop"] for item in support.values()) != metadata["sample_counts"]["test_points"]
            or sum(item["window_samples"] for item in support.values()) != metadata["sample_counts"]["test_samples"]):
        raise ValueError(f"{spec.method_name} window counts and score support are inconsistent")
    epoch_limit = configuration["Model_Params"].get("epochs", configuration["Model_Params"].get("num_epochs")) - int(spec.method_name in ("LSTMADalpha", "LSTMADbeta"))
    if metadata.get("selection_status") == "available" and (type(metadata.get("best_epoch")) is not int or not 1 <= metadata["best_epoch"] <= epoch_limit):
        raise ValueError(f"{spec.method_name} selected epoch is outside the native epoch lifecycle")


__all__ = [
    "ScoreSource",
    "load_model_artifact",
    "resolve_selection_status",
    "save_fitted_estimator",
    "score_source_for_execution",
    "validate_score_lineage",
]
