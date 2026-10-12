from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Any

from timesigma.algorithms import EASYTSAD_REVISION, EASYTSAD_SCHEMAS
from timesigma.benchmark.leaderboard import OFFICIAL_METRIC_IDS
from timesigma.benchmark.zero_shot import ZeroShotSplit
from timesigma.profiles.features import CURVE_PROFILE_DEFINITION

from .decision import ResearchDecision
from .pipeline import (
    NextAction,
    NormalizedPipeline,
    PlanningUnit,
)


EVIDENCE_KINDS = ("component", "research_step")
ACTION_DIMENSIONS = (
    "algorithm",
    "preprocess",
    "diff_order",
    "window",
    "hyperparameter",
    "preprocessing",
    "ensemble",
)


@dataclass(frozen=True)
class ComponentEvidence:
    evidence_id: str
    task: Mapping[str, Any]
    pipeline: Mapping[str, Any]
    metrics: Mapping[str, float]
    artifact: Mapping[str, Any]
    kind: str = "component"

    def __post_init__(self) -> None:
        _require_text(self.evidence_id, "evidence_id")
        if self.kind != "component":
            raise ValueError("ComponentEvidence kind must be component")
        task = _task_payload(self.task)
        pipeline = _pipeline_payload(self.pipeline)
        expected_id = component_evidence_id(
            method=str(pipeline["method"]),
            task=task,
        )
        if self.evidence_id != expected_id:
            raise ValueError("ComponentEvidence evidence_id does not match its task")
        object.__setattr__(self, "task", _freeze_mapping(task))
        object.__setattr__(self, "pipeline", _freeze_mapping(pipeline))
        object.__setattr__(
            self,
            "metrics",
            _freeze_mapping(_metrics_payload(self.metrics)),
        )
        object.__setattr__(
            self,
            "artifact",
            _freeze_mapping(_artifact_payload(self.artifact)),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ComponentEvidence:
        _exact_fields(
            payload,
            {"evidence_id", "kind", "task", "pipeline", "metrics", "artifact"},
            "ComponentEvidence",
        )
        return cls(
            evidence_id=_string(payload, "evidence_id"),
            kind=_string(payload, "kind"),
            task=_mapping(payload, "task"),
            pipeline=_mapping(payload, "pipeline"),
            metrics=_mapping(payload, "metrics"),
            artifact=_mapping(payload, "artifact"),
        )

    def canonical_view(self) -> dict[str, Any]:
        return {**self.as_dict(), "pipeline": NormalizedPipeline.from_dict(self.pipeline).canonical_view(str(self.task["schema"]))}

    @classmethod
    def from_canonical_view(cls, payload: Mapping[str, Any]) -> ComponentEvidence:
        pipeline = NormalizedPipeline.from_canonical_view(payload["pipeline"], str(payload["task"]["schema"]))
        return cls.from_dict({**payload, "pipeline": pipeline.canonical_payload()})

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "task": _thaw(self.task),
            "pipeline": _thaw(self.pipeline),
            "metrics": _thaw(self.metrics),
            "artifact": _thaw(self.artifact),
        }


@dataclass(frozen=True)
class ResearchStepEvidence:
    evidence_id: str
    task: Mapping[str, Any]
    target_metric: str
    baseline_evidence_id: str
    baseline_pipeline: Mapping[str, Any]
    candidate_pipeline: Mapping[str, Any]
    action: Mapping[str, Any]
    before_value: float
    after_value: float
    edge_delta: float
    metrics: Mapping[str, float]
    research_decision: Mapping[str, Any]
    artifact: Mapping[str, Any]
    kind: str = "research_step"

    def __post_init__(self) -> None:
        _require_text(self.evidence_id, "evidence_id")
        if self.kind != "research_step":
            raise ValueError("ResearchStepEvidence kind must be research_step")
        target_metric = _official_metric(self.target_metric, "target_metric")
        baseline_evidence_id = _require_text(
            self.baseline_evidence_id,
            "baseline_evidence_id",
        )
        before = _finite_float(self.before_value, "before_value")
        after = _finite_float(self.after_value, "after_value")
        delta = _finite_float(self.edge_delta, "edge_delta")
        if not math.isclose(delta, after - before, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("edge_delta must equal after_value - before_value")

        task = _task_payload(self.task)
        baseline = _pipeline_payload(self.baseline_pipeline)
        candidate = _pipeline_payload(self.candidate_pipeline)
        action = _action_payload(self.action)
        _validate_research_step_action(
            task=task,
            target_metric=target_metric,
            baseline_pipeline=baseline,
            candidate_pipeline=candidate,
            action=action,
        )
        _validate_research_step_evidence_id(self.evidence_id, task)

        object.__setattr__(self, "target_metric", target_metric)
        object.__setattr__(self, "baseline_evidence_id", baseline_evidence_id)
        object.__setattr__(self, "before_value", before)
        object.__setattr__(self, "after_value", after)
        object.__setattr__(self, "edge_delta", delta)
        object.__setattr__(self, "task", _freeze_mapping(task))
        object.__setattr__(self, "baseline_pipeline", _freeze_mapping(baseline))
        object.__setattr__(self, "candidate_pipeline", _freeze_mapping(candidate))
        object.__setattr__(
            self,
            "action",
            _freeze_mapping(action),
        )
        object.__setattr__(
            self,
            "metrics",
            _freeze_mapping(_metrics_payload(self.metrics)),
        )
        object.__setattr__(
            self,
            "research_decision",
            _freeze_mapping(_research_decision_payload(self.research_decision)),
        )
        object.__setattr__(
            self,
            "artifact",
            _freeze_mapping(_artifact_payload(self.artifact)),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResearchStepEvidence:
        _exact_fields(
            payload,
            {
                "evidence_id",
                "kind",
                "task",
                "target_metric",
                "baseline_evidence_id",
                "baseline_pipeline",
                "candidate_pipeline",
                "action",
                "before_value",
                "after_value",
                "edge_delta",
                "metrics",
                "research_decision",
                "artifact",
            },
            "ResearchStepEvidence",
        )
        return cls(
            evidence_id=_string(payload, "evidence_id"),
            kind=_string(payload, "kind"),
            task=_mapping(payload, "task"),
            target_metric=_string(payload, "target_metric"),
            baseline_evidence_id=_string(payload, "baseline_evidence_id"),
            baseline_pipeline=_mapping(payload, "baseline_pipeline"),
            candidate_pipeline=_mapping(payload, "candidate_pipeline"),
            action=_mapping(payload, "action"),
            before_value=_finite_float(payload.get("before_value"), "before_value"),
            after_value=_finite_float(payload.get("after_value"), "after_value"),
            edge_delta=_finite_float(payload.get("edge_delta"), "edge_delta"),
            metrics=_mapping(payload, "metrics"),
            research_decision=_mapping(payload, "research_decision"),
            artifact=_mapping(payload, "artifact"),
        )

    def canonical_view(self) -> dict[str, Any]:
        return {**self.as_dict(), **{
            name: NormalizedPipeline.from_dict(getattr(self, name)).canonical_view(str(self.task["schema"]))
            for name in ("baseline_pipeline", "candidate_pipeline")
        }}

    @classmethod
    def from_canonical_view(cls, payload: Mapping[str, Any]) -> ResearchStepEvidence:
        return cls.from_dict({**payload, **{
            name: NormalizedPipeline.from_canonical_view(payload[name], str(payload["task"]["schema"])).canonical_payload()
            for name in ("baseline_pipeline", "candidate_pipeline")
        }})

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "task": _thaw(self.task),
            "target_metric": self.target_metric,
            "baseline_evidence_id": self.baseline_evidence_id,
            "baseline_pipeline": _thaw(self.baseline_pipeline),
            "candidate_pipeline": _thaw(self.candidate_pipeline),
            "action": _thaw(self.action),
            "before_value": self.before_value,
            "after_value": self.after_value,
            "edge_delta": self.edge_delta,
            "metrics": _thaw(self.metrics),
            "research_decision": _thaw(self.research_decision),
            "artifact": _thaw(self.artifact),
        }


def agent_memory_manifest_payload(
    *,
    dataset_source: str | None = None,
    dataset_sources: Mapping[str, str] | None = None,
    dataset_revisions: Mapping[str, str],
    seed: int = 1,
    easytsad_revision: str = EASYTSAD_REVISION,
    profile_definition: str = CURVE_PROFILE_DEFINITION,
    zero_shot_membership: str | None = "zero-shot-membership.json",
) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return {
        "store": "agent-memory",
        "easytsad_revision": _require_text(easytsad_revision, "easytsad_revision"),
        **(
            {}
            if dataset_source is None
            else {"dataset_source": _require_text(dataset_source, "dataset_source")}
        ),
        **(
            {}
            if dataset_sources is None
            else {
                "dataset_sources": _string_key_text_mapping(
                    dataset_sources,
                    "dataset_sources",
                )
            }
        ),
        "dataset_revisions": _string_key_text_mapping(
            dataset_revisions,
            "dataset_revisions",
        ),
        "seed": seed,
        "profile_definition": _require_text(profile_definition, "profile_definition"),
        "zero_shot_membership": _relative_reference(
            zero_shot_membership,
            "zero_shot_membership",
            allow_none=True,
        ),
        "execution_assumptions": {
            "pipeline_start": "schema_raw_train_values",
            "preprocess_and_diff_order": "pipeline_facts",
        },
    }


def zero_shot_membership_payload(
    splits: Sequence[ZeroShotSplit | Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "zero_shot",
        "dataset_type": "UTS",
        "datasets": [_zero_shot_split_payload(split) for split in splits],
    }


def research_step_action_payload(
    *,
    baseline_pipeline: Mapping[str, Any],
    candidate_pipeline: Mapping[str, Any],
    action: NextAction,
) -> dict[str, Any]:
    return _edge_action(
        baseline_pipeline=NormalizedPipeline.from_dict(baseline_pipeline),
        candidate_pipeline=NormalizedPipeline.from_dict(candidate_pipeline),
        action=action,
    )


def component_evidence_id(
    *,
    method: str,
    task: Mapping[str, Any],
    configuration: str | None = None,
) -> str:
    _ = configuration
    payload = _task_payload(task)
    return "/".join(
        (
            "component",
            _identity_segment(method),
            _identity_segment(str(payload["schema"])),
            _identity_segment(str(payload["dataset"])),
            planning_unit_identity(str(payload["schema"]), payload["unit"]),
        )
    )


def research_step_evidence_id(
    *,
    task: Mapping[str, Any],
    experiment_id: str,
) -> str:
    payload = _task_payload(task)
    return "/".join(
        (
            "research-step",
            _identity_segment(str(payload["dataset"])),
            _identity_segment(str(payload["schema"])),
            planning_unit_identity(str(payload["schema"]), payload["unit"]),
            _identity_segment(experiment_id),
        )
    )


def planning_unit_identity(schema: str, payload: Mapping[str, Any]) -> str:
    unit = PlanningUnit.from_dict(schema, payload)
    if schema == "naive":
        assert unit.curve is not None
        return _identity_segment(unit.curve)
    if schema == "all_in_one":
        return "dataset"
    if schema == "zero_shot":
        return "split"
    raise ValueError(f"unsupported EasyTSAD schema: {schema}")


def _validate_research_step_evidence_id(
    evidence_id: str,
    task: Mapping[str, Any],
) -> None:
    prefix = research_step_evidence_id(task=task, experiment_id="placeholder")
    expected_prefix = prefix.rsplit("/", 1)[0] + "/"
    if not evidence_id.startswith(expected_prefix):
        raise ValueError("ResearchStepEvidence evidence_id does not match its task")
    experiment_segment = evidence_id.removeprefix(expected_prefix)
    if experiment_segment != _identity_segment(experiment_segment):
        raise ValueError("ResearchStepEvidence evidence_id has an invalid experiment id")


def _validate_research_step_action(
    *,
    task: Mapping[str, Any],
    target_metric: str,
    baseline_pipeline: Mapping[str, Any],
    candidate_pipeline: Mapping[str, Any],
    action: Mapping[str, Any],
) -> None:
    next_action = _next_action_from_edge(action)
    # Stored records include historical bundled comparisons. New execution and
    # commit admission enforce semantic control; loading verifies recorded facts.
    before = NormalizedPipeline.from_dict(baseline_pipeline).canonical_payload()
    after = NormalizedPipeline.from_dict(candidate_pipeline).canonical_payload()
    expected_pipeline = deepcopy(before)
    dimension = next_action.dimension
    if dimension == "algorithm":
        if before["method"] == after["method"]:
            raise ValueError("algorithm action requires a detector change")
        expected_pipeline["method"] = after["method"]
        expected_pipeline["resolved_configuration"] = deepcopy(after["resolved_configuration"])
    elif dimension in {"preprocessing", "ensemble"}:
        field = "operators" if dimension == "preprocessing" else "ensemble_ref"
        expected_pipeline[field] = deepcopy(after[field])
    else:
        section = "Data_Params" if dimension in {"preprocess", "diff_order"} else "Model_Params"
        parameter = dimension if section == "Data_Params" else str(action["parameter"])
        expected_pipeline["resolved_configuration"][section][parameter] = deepcopy(after["resolved_configuration"][section][parameter])
    if before == after:
        raise ValueError("ResearchStepEvidence candidate must differ from its baseline")
    if expected_pipeline != after:
        raise ValueError("ResearchStepEvidence action does not match its recorded Pipeline transition")
    expected = _edge_action(
        baseline_pipeline=NormalizedPipeline.from_dict(baseline_pipeline),
        candidate_pipeline=NormalizedPipeline.from_dict(candidate_pipeline),
        action=next_action,
    )
    if expected != dict(action):
        raise ValueError("ResearchStepEvidence action does not match its Pipeline edge")


def _next_action_from_edge(action: Mapping[str, Any]) -> NextAction:
    dimension = str(action["dimension"])
    if dimension == "ensemble" and action["after"] is None:
        return NextAction("ensemble", value={"method": "none"})
    if dimension == "hyperparameter":
        return NextAction(
            "hyperparameter",
            parameter=str(action["parameter"]),
            value=deepcopy(action["after"]),
        )
    if dimension == "window":
        return NextAction("window", value=deepcopy(action["after"]))
    return NextAction(
        dimension,  # type: ignore[arg-type]
        value=deepcopy(action["after"]),
    )


def _edge_action(
    *,
    baseline_pipeline: NormalizedPipeline,
    candidate_pipeline: NormalizedPipeline,
    action: NextAction,
) -> dict[str, Any]:
    baseline = baseline_pipeline.canonical_payload()
    candidate = candidate_pipeline.canonical_payload()
    dimension = action.dimension
    payload: dict[str, Any] = {"dimension": dimension}
    if dimension == "algorithm":
        before = baseline["method"]
        after = candidate["method"]
    elif dimension == "preprocess":
        before = baseline["resolved_configuration"]["Data_Params"]["preprocess"]
        after = candidate["resolved_configuration"]["Data_Params"]["preprocess"]
    elif dimension == "diff_order":
        before = baseline["resolved_configuration"]["Data_Params"]["diff_order"]
        after = candidate["resolved_configuration"]["Data_Params"]["diff_order"]
    elif dimension == "window":
        from .semantics import context_window_parameter

        parameter = context_window_parameter(baseline_pipeline.method)
        if parameter is None:
            raise ValueError("window action requires a Catalog window parameter")
        payload["parameter"] = parameter
        before = baseline["resolved_configuration"]["Model_Params"][parameter]
        after = candidate["resolved_configuration"]["Model_Params"][parameter]
    elif dimension == "hyperparameter":
        if action.parameter is None:
            raise ValueError("hyperparameter action requires a parameter")
        payload["parameter"] = action.parameter
        before = baseline["resolved_configuration"]["Model_Params"][action.parameter]
        after = candidate["resolved_configuration"]["Model_Params"][action.parameter]
    elif dimension in {"preprocessing", "ensemble"}:
        field = "operators" if dimension == "preprocessing" else "ensemble_ref"
        before = baseline[field]
        after = candidate[field]
    else:
        raise ValueError(f"unsupported executable action dimension: {dimension}")
    payload["before"] = deepcopy(before)
    payload["after"] = deepcopy(after)
    return payload


def _decision_payload(decision: ResearchDecision) -> dict[str, Any]:
    payload = decision.as_dict()
    if decision.next_action_rationale is None:
        payload["next_action_rationale"] = None
    return payload


def _next_action(payload: Mapping[str, Any]) -> NextAction:
    return NextAction(
        dimension=_string(payload, "dimension"),  # type: ignore[arg-type]
        parameter=_optional_string(payload.get("parameter"), "parameter"),
        value=deepcopy(payload.get("value")),
    )


def _task_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(payload, {"dataset", "schema", "unit"}, "task")
    dataset = _string(payload, "dataset")
    if dataset.casefold() == "aiops":
        raise ValueError("AIOPS cannot become persistent canonical Evidence")
    schema = _string(payload, "schema")
    if schema not in EASYTSAD_SCHEMAS:
        raise ValueError(f"unsupported EasyTSAD schema: {schema}")
    unit = PlanningUnit.from_dict(schema, _mapping(payload, "unit"))
    return {
        "dataset": dataset,
        "schema": schema,
        "unit": unit.as_dict(),
    }


def _action_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        {"dimension", "parameter", "before", "after"}
        if "parameter" in payload
        else {"dimension", "before", "after"}
    )
    _exact_fields(payload, fields, "action")
    dimension = _string(payload, "dimension")
    if dimension not in ACTION_DIMENSIONS:
        raise ValueError(f"unsupported canonical executable action dimension: {dimension}")
    parameter = _optional_string(payload.get("parameter"), "parameter")
    if dimension in {"window", "hyperparameter"} and parameter is None:
        raise ValueError(f"{dimension} action requires a parameter")
    if dimension not in {"window", "hyperparameter"} and parameter is not None:
        raise ValueError(f"{dimension} action does not accept a parameter")
    action: dict[str, Any] = {"dimension": dimension}
    if parameter is not None:
        action["parameter"] = parameter
    action["before"] = deepcopy(payload.get("before"))
    action["after"] = deepcopy(payload.get("after"))
    return action


def _research_decision_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _decision_payload(ResearchDecision.from_dict(payload))


def _pipeline_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return NormalizedPipeline.from_dict(payload).canonical_payload()


def _artifact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(
        payload,
        {
            "artifact_type",
            "selection_status",
            "score_source",
            "artifact_path",
        },
        "artifact",
    )
    artifact_type = _string(payload, "artifact_type")
    if artifact_type not in {"torch_state_dict", "fitted_estimator", "replay_recipe", "tensorflow_checkpoint"}:
        raise ValueError("unsupported artifact_type")
    selection_status = _string(payload, "selection_status")
    if selection_status not in {"available", "unavailable", "not_applicable"}:
        raise ValueError("unsupported selection_status")
    return {
        "artifact_type": artifact_type,
        "selection_status": selection_status,
        "score_source": _string(payload, "score_source"),
        "artifact_path": _relative_reference(
            payload.get("artifact_path"),
            "artifact_path",
            allow_none=True,
        ),
    }


def _artifact_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _artifact_payload(
        {
            "artifact_type": payload.get("artifact_type"),
            "selection_status": payload.get("selection_status"),
            "score_source": payload.get("score_source"),
            "artifact_path": payload.get("state"),
        }
    )


def _metrics_payload(payload: Mapping[str, Any]) -> dict[str, float]:
    if set(payload) != set(OFFICIAL_METRIC_IDS):
        raise ValueError("canonical Evidence requires the complete EasyTSAD metric vector")
    return {
        metric_id: _finite_float(payload[metric_id], metric_id)
        for metric_id in OFFICIAL_METRIC_IDS
    }


def _zero_shot_split_payload(split: ZeroShotSplit | Mapping[str, Any]) -> dict[str, Any]:
    payload = split.as_dict() if isinstance(split, ZeroShotSplit) else dict(split)
    fields = {
        "dataset",
        "proportion",
        "random_seed",
        "enumerated_curves",
        "training_curves",
        "held_out_curves",
    }
    _exact_fields(payload, fields, "zero-shot membership split")
    dataset = _require_text(payload["dataset"], "dataset")
    proportion = _finite_float(payload["proportion"], "proportion")
    if not 0.0 < proportion < 1.0:
        raise ValueError("zero-shot split proportion must be between 0 and 1")
    seed = payload["random_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("zero-shot split random_seed must be an integer")
    enumerated = _string_sequence(payload["enumerated_curves"], "enumerated_curves")
    training = _string_sequence(payload["training_curves"], "training_curves")
    held_out = _string_sequence(payload["held_out_curves"], "held_out_curves")
    if set(training) & set(held_out):
        raise ValueError("zero-shot training and held-out members must be disjoint")
    if set(training) | set(held_out) != set(enumerated):
        raise ValueError("zero-shot training and held-out members must partition enumerated members")
    return {
        "dataset": dataset,
        "proportion": proportion,
        "random_seed": seed,
        "enumerated_curves": list(enumerated),
        "training_curves": list(training),
        "held_out_curves": list(held_out),
    }


def _curve_names(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    return _string_sequence(value, f"profile {key}")


def _string_sequence(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a sequence")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise TypeError(f"{name} must contain non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must contain unique strings")
    return tuple(value)


def _string_key_text_mapping(value: Mapping[str, Any], name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        result[key] = _require_text(item, f"{name}.{key}")
    return result


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{key} must be a string-keyed mapping")
    return value


def _string(payload: Mapping[str, Any], key: str) -> str:
    return _require_text(payload.get(key), key)


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, name)


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _official_metric(value: Any, name: str) -> str:
    metric = _require_text(value, name)
    if metric not in OFFICIAL_METRIC_IDS:
        raise ValueError(f"unsupported EasyTSAD official metric: {metric}")
    return metric


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _relative_reference(
    value: Any,
    name: str,
    *,
    allow_none: bool,
) -> str | None:
    if value is None and allow_none:
        return None
    reference = _require_text(value, name)
    path = PurePosixPath(reference)
    if path.is_absolute() or ".." in path.parts or "://" in reference:
        raise ValueError(f"{name} must be a relative path")
    return path.as_posix()


def _identity_segment(value: str) -> str:
    text = value.strip().replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    if not text:
        raise ValueError("identity segment cannot be empty")
    return text


def _exact_fields(payload: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(payload) != fields:
        raise ValueError(f"{name} fields do not match canonical Evidence contract")


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze(item) for key, item in value.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return deepcopy(value)


__all__ = [
    "ComponentEvidence",
    "ResearchStepEvidence",
    "agent_memory_manifest_payload",
    "component_evidence_id",
    "planning_unit_identity",
    "research_step_action_payload",
    "research_step_evidence_id",
    "zero_shot_membership_payload",
]
