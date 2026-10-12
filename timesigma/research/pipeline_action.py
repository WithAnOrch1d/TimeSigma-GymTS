from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import json
from types import MappingProxyType
from typing import Any, Literal

from .pipeline import PipelineSpec, TaskContext
from .pipeline_space import (
    PipelineDefinition,
    pipeline_component_registry,
    validate_pipeline_definition,
)


PipelineActionOperation = Literal["add_component"]
PipelineActionStage = Literal["preprocessing"]


@dataclass(frozen=True)
class PipelineAction:
    operation: PipelineActionOperation
    stage: PipelineActionStage
    component_id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.operation != "add_component":
            raise ValueError("PipelineAction MVP only supports add_component")
        if self.stage != "preprocessing":
            raise ValueError("PipelineAction MVP only supports preprocessing")
        if not self.component_id.strip():
            raise ValueError("PipelineAction component_id is required")
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(deepcopy(dict(self.parameters))),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PipelineAction:
        fields = {"operation", "stage", "component_id", "parameters"}
        unknown = sorted(set(payload) - fields)
        missing = sorted(fields - {"parameters"} - set(payload))
        if unknown:
            raise ValueError(f"Unknown PipelineAction field: {unknown[0]}")
        if missing:
            raise ValueError(f"Missing PipelineAction field: {missing[0]}")
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise TypeError("PipelineAction parameters must be a mapping")
        return cls(
            operation=_text(payload.get("operation"), "operation"),  # type: ignore[arg-type]
            stage=_text(payload.get("stage"), "stage"),  # type: ignore[arg-type]
            component_id=_text(payload.get("component_id"), "component_id"),
            parameters=parameters,
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "operation": self.operation,
            "stage": self.stage,
            "component_id": self.component_id,
        }
        if self.parameters:
            payload["parameters"] = _plain_json(self.parameters)
        return payload


@dataclass(frozen=True)
class PipelineDefinitionDiff:
    changed_dimensions: tuple[str, ...]
    changed_components: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "changed_dimensions",
            tuple(self.changed_dimensions),
        )
        object.__setattr__(
            self,
            "changed_components",
            tuple(_freeze_mapping(dict(item)) for item in self.changed_components),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "changed_dimensions": list(self.changed_dimensions),
            "changed_components": [_plain_json(item) for item in self.changed_components],
        }


def apply_pipeline_action(
    baseline_pipeline: PipelineDefinition | PipelineSpec | Mapping[str, Any],
    action: PipelineAction,
    task: TaskContext | None = None,
) -> PipelineDefinition:
    baseline = _pipeline_definition(baseline_pipeline)
    component = _component_ref(action, task)
    if any(
        item.get("component_id") == action.component_id
        for item in baseline.preprocessing
    ):
        raise ValueError("preprocessing component already exists in the baseline")
    candidate = PipelineDefinition(
        preprocessing=(*baseline.preprocessing, component),
        representation=baseline.representation,
        detector=baseline.detector,
        postprocessing=baseline.postprocessing,
        composition=baseline.composition,
    )
    validate_pipeline_action_change(baseline, action, candidate, task)
    return candidate


def validate_pipeline_action_change(
    baseline_pipeline: PipelineDefinition | PipelineSpec | Mapping[str, Any],
    action: PipelineAction,
    candidate_pipeline: PipelineDefinition | PipelineSpec | Mapping[str, Any],
    task: TaskContext | None = None,
) -> str:
    baseline = _pipeline_definition(baseline_pipeline)
    candidate = _pipeline_definition(candidate_pipeline)
    _component_ref(action, task)
    if task is not None:
        validate_pipeline_definition(baseline, task)
        validate_pipeline_definition(candidate, task)

    diff = pipeline_definition_diff(baseline, candidate)
    if not diff.changed_dimensions:
        raise ValueError("Candidate PipelineDefinition must differ from the baseline")
    if diff.changed_dimensions != ("preprocessing",):
        raise ValueError(
            "Candidate PipelineDefinition must change exactly one research dimension: preprocessing"
        )
    expected_preprocessing = (*baseline.preprocessing, _component_ref(action, task))
    if _canonical(candidate.preprocessing) != _canonical(expected_preprocessing):
        raise ValueError("Candidate PipelineDefinition does not match the PipelineAction")
    if _canonical(candidate.detector) != _canonical(baseline.detector):
        raise ValueError("PipelineAction MVP must keep detector unchanged")
    return "preprocessing"


def pipeline_definition_diff(
    baseline_pipeline: PipelineDefinition | PipelineSpec | Mapping[str, Any],
    candidate_pipeline: PipelineDefinition | PipelineSpec | Mapping[str, Any],
) -> PipelineDefinitionDiff:
    baseline = _pipeline_definition(baseline_pipeline)
    candidate = _pipeline_definition(candidate_pipeline)
    changes: list[tuple[str, Mapping[str, Any]]] = []
    for dimension, before, after in (
        ("preprocessing", baseline.preprocessing, candidate.preprocessing),
        ("representation", baseline.representation, candidate.representation),
        ("detector", baseline.detector, candidate.detector),
        ("postprocessing", baseline.postprocessing, candidate.postprocessing),
        ("pipeline_topology", baseline.composition, candidate.composition),
    ):
        if _canonical(before) != _canonical(after):
            changes.append(
                (
                    dimension,
                    {
                        "dimension": dimension,
                        "before": _plain_json(before),
                        "after": _plain_json(after),
                    },
                )
            )
    return PipelineDefinitionDiff(
        changed_dimensions=tuple(dimension for dimension, _ in changes),
        changed_components=tuple(component for _, component in changes),
    )


def _pipeline_definition(
    pipeline: PipelineDefinition | PipelineSpec | Mapping[str, Any],
) -> PipelineDefinition:
    if isinstance(pipeline, PipelineDefinition):
        return pipeline
    if isinstance(pipeline, PipelineSpec):
        return PipelineDefinition.from_pipeline_spec(pipeline)
    if "preprocessing" in pipeline or "detector" in pipeline or "composition" in pipeline:
        return PipelineDefinition.from_dict(pipeline)
    method = pipeline.get("method")
    configuration = pipeline.get("resolved_configuration")
    if not isinstance(method, str) or not isinstance(configuration, Mapping):
        raise TypeError("Pipeline mapping requires method and resolved_configuration")
    return PipelineDefinition.from_pipeline_spec(
        PipelineSpec(method=method, resolved_configuration=configuration),
    )


def _component_ref(
    action: PipelineAction,
    task: TaskContext | None,
) -> Mapping[str, Any]:
    registry = pipeline_component_registry()
    component = registry.get(action.component_id)
    if component is None:
        raise ValueError(f"unknown Pipeline component: {action.component_id}")
    if component.stage != "preprocessing":
        raise ValueError("PipelineAction MVP only accepts preprocessing components")
    if component.execution_adapter != "materialized_dataset_view":
        raise ValueError("PipelineAction MVP only accepts materialized preprocessing components")
    if not component.label_free:
        raise ValueError("PipelineAction MVP only accepts label-free components")
    if task is not None and not component.supports(task.schema):
        raise ValueError(f"Pipeline component {action.component_id} is not executable for {task.schema}")
    return MappingProxyType(
        {
            "component_id": action.component_id,
            "parameters": _freeze_mapping(dict(action.parameters)),
        }
    )


def _canonical(value: Any) -> str:
    return json.dumps(
        _plain_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    if isinstance(value, list):
        return [_plain_json(item) for item in value]
    return deepcopy(value)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not all(isinstance(key, str) for key in value):
        raise TypeError("PipelineAction parameters must use string keys")
    return MappingProxyType({key: _freeze(item) for key, item in value.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return deepcopy(value)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"PipelineAction {label} must be a non-empty string")
    return value


__all__ = [
    "PipelineAction",
    "PipelineDefinitionDiff",
    "apply_pipeline_action",
    "pipeline_definition_diff",
    "validate_pipeline_action_change",
]
