from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

from timesigma.profiles.features import PROFILE_FEATURES, CurveProfile
from timesigma.profiles.store import load_curve_profile_names, load_curve_profiles

from .canonical import ComponentEvidence, planning_unit_identity
from .collection import EvidenceCollection, plain_json
from .pipeline import TaskContext


PROFILE_VECTOR_FIELDS = ("point_count", *PROFILE_FEATURES)
DEFAULT_HISTORY_LIMIT = 4
DEFAULT_MINIMUM_SIMILARITY = 0.75


def retrieve_component_history(
    *,
    collection: EvidenceCollection,
    target_task: TaskContext,
    target_profile: Mapping[str, Any],
    agent_memory_root: Path,
    target_metric: str | None = None,
    limit: int = DEFAULT_HISTORY_LIMIT,
    minimum_similarity: float = DEFAULT_MINIMUM_SIMILARITY,
) -> tuple[dict[str, Any], ...]:
    if limit < 1:
        raise ValueError("history retrieval limit must be positive")
    if not math.isfinite(minimum_similarity):
        raise ValueError("minimum similarity must be finite")

    target_vector = _profile_vector(target_profile)
    if not target_vector:
        return ()

    manifest = _memory_manifest(agent_memory_root)
    if manifest is None:
        return ()
    profile_cache = _ProfileCache(agent_memory_root.parent / "profiles", manifest)
    metric_id = target_metric or target_task.target_metric
    candidates: list[tuple[float, float, ComponentEvidence]] = []

    for evidence in collection.evidence:
        if not isinstance(evidence, ComponentEvidence):
            continue
        if _same_task(evidence.task, target_task):
            continue
        if str(evidence.task["dataset"]).casefold() == "aiops":
            continue
        source_vector = profile_cache.vector_for_task(evidence.task)
        if not source_vector:
            continue
        similarity, distance = _similarity(target_vector, source_vector)
        if similarity < minimum_similarity:
            continue
        candidates.append((similarity, distance, evidence))

    ordered = sorted(
        candidates,
        key=lambda item: (
            -item[0],
            -_metric_value(item[2], metric_id),
            item[2].evidence_id,
        ),
    )
    return tuple(
        _history_item(
            evidence=evidence,
            metric_id=metric_id,
            similarity=similarity,
            distance=distance,
        )
        for similarity, distance, evidence in ordered[:limit]
    )


class _ProfileCache:
    def __init__(self, profile_root: Path, manifest: Mapping[str, Any]) -> None:
        self._profile_root = profile_root
        self._sources = _dataset_sources(manifest)
        self._revisions = _string_mapping(manifest.get("dataset_revisions"))
        self._profiles: dict[str, dict[str, CurveProfile]] = {}

    def vector_for_task(self, task: Mapping[str, Any]) -> dict[str, float] | None:
        dataset = str(task["dataset"])
        profiles = self._profiles_for_dataset(dataset)
        if profiles is None:
            return None
        unit = task.get("unit")
        if not isinstance(unit, Mapping):
            return None
        training_curves = unit.get("training_curves")
        if not isinstance(training_curves, (list, tuple)) or not all(
            isinstance(curve, str) for curve in training_curves
        ):
            return None
        selected = [profiles[curve] for curve in training_curves if curve in profiles]
        if len(selected) != len(training_curves) or not selected:
            return None
        return _curve_profiles_vector(tuple(selected))

    def _profiles_for_dataset(self, dataset: str) -> dict[str, CurveProfile] | None:
        if dataset in self._profiles:
            return self._profiles[dataset]
        source = self._sources.get(dataset)
        revision = self._revisions.get(dataset)
        if source is None or revision is None:
            return None
        profile_dir = self._profile_root / dataset / revision
        if not profile_dir.is_dir():
            return None
        provenance = {"source": source, "revision": revision}
        curve_names = load_curve_profile_names(
            dataset=dataset,
            profile_dir=profile_dir,
            provenance=provenance,
        )
        profiles = load_curve_profiles(
            dataset=dataset,
            profile_dir=profile_dir,
            provenance=provenance,
            curve_names=curve_names,
        )
        self._profiles[dataset] = profiles
        return profiles


def _memory_manifest(agent_memory_root: Path) -> Mapping[str, Any] | None:
    path = agent_memory_root / "manifest.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("agent-memory manifest must be a JSON object")
    return payload


def _dataset_sources(manifest: Mapping[str, Any]) -> dict[str, str]:
    sources = _string_mapping(manifest.get("dataset_sources"))
    if sources:
        return sources
    source = manifest.get("dataset_source")
    if isinstance(source, str):
        return {
            dataset: source
            for dataset in _string_mapping(manifest.get("dataset_revisions"))
        }
    return {}


def _string_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, str)
    }


def _profile_vector(profile: Mapping[str, Any]) -> dict[str, float]:
    curve_profile = profile.get("curve_profile")
    if isinstance(curve_profile, Mapping):
        return _facts_vector(curve_profile)
    aggregate = profile.get("aggregate")
    if isinstance(aggregate, Mapping):
        return _aggregate_vector(aggregate)
    return {}


def _curve_profiles_vector(profiles: tuple[CurveProfile, ...]) -> dict[str, float]:
    vector: dict[str, float] = {}
    for field in PROFILE_VECTOR_FIELDS:
        values = [
            float(value)
            for profile in profiles
            if (value := getattr(profile, field)) is not None
        ]
        if values:
            vector[field] = sum(values) / len(values)
    return vector


def _facts_vector(facts: Mapping[str, Any]) -> dict[str, float]:
    vector: dict[str, float] = {}
    for field in PROFILE_VECTOR_FIELDS:
        value = facts.get(field)
        if _finite_number(value):
            vector[field] = float(value)
    return vector


def _aggregate_vector(aggregate: Mapping[str, Any]) -> dict[str, float]:
    features = aggregate.get("features")
    if not isinstance(features, Mapping):
        return {}
    vector: dict[str, float] = {}
    for field in PROFILE_VECTOR_FIELDS:
        summary = features.get(field)
        if not isinstance(summary, Mapping):
            continue
        value = summary.get("mean")
        if _finite_number(value):
            vector[field] = float(value)
    return vector


def _similarity(
    target: Mapping[str, float],
    source: Mapping[str, float],
) -> tuple[float, float]:
    diffs: list[float] = []
    for field in PROFILE_VECTOR_FIELDS:
        target_value = target.get(field)
        source_value = source.get(field)
        if target_value is None and source_value is None:
            continue
        if target_value is None or source_value is None:
            diffs.append(1.0)
            continue
        diffs.append(abs(_scale(target_value) - _scale(source_value)))
    if not diffs:
        return 0.0, math.inf
    distance = math.sqrt(sum(diff * diff for diff in diffs) / len(diffs))
    return 1.0 / (1.0 + distance), distance


def _scale(value: float) -> float:
    magnitude = math.log1p(abs(value))
    return -magnitude if value < 0.0 else magnitude


def _history_item(
    *,
    evidence: ComponentEvidence,
    metric_id: str,
    similarity: float,
    distance: float,
) -> dict[str, Any]:
    task = evidence.task
    unit = task["unit"]
    pipeline = evidence.pipeline
    configuration = pipeline["resolved_configuration"]
    return {
        "evidence_id": evidence.evidence_id,
        "task_identity": {
            "dataset": str(task["dataset"]),
            "schema": str(task["schema"]),
            "planning_unit_id": planning_unit_identity(str(task["schema"]), unit),
            "training_member_count": len(unit["training_curves"]),
            "held_out_member_count": len(unit["held_out_curves"]),
        },
        "similarity_score": round(similarity, 6),
        "profile_distance": round(distance, 6),
        "historical_pipeline": {
            "method": str(pipeline["method"]),
            "data_params": plain_json(configuration["Data_Params"]),
            "model_params": plain_json(configuration["Model_Params"]),
        },
        "metric_result": {
            "target_metric": metric_id,
            "value": _metric_value(evidence, metric_id),
        },
    }


def _metric_value(evidence: ComponentEvidence, metric_id: str) -> float:
    value = evidence.metrics.get(metric_id)
    if not _finite_number(value):
        raise ValueError(f"ComponentEvidence is missing target metric: {evidence.evidence_id}")
    return float(value)


def _same_task(task: Mapping[str, Any], target: TaskContext) -> bool:
    return (
        task.get("dataset") == target.dataset
        and task.get("schema") == target.schema
        and plain_json(task.get("unit")) == plain_json(target.unit.as_dict())
    )


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


__all__ = [
    "DEFAULT_HISTORY_LIMIT",
    "DEFAULT_MINIMUM_SIMILARITY",
    "PROFILE_VECTOR_FIELDS",
    "retrieve_component_history",
]
