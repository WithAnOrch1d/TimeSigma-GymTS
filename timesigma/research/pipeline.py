from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

from timesigma.algorithms import AlgorithmSpec, EASYTSAD_SCHEMAS, get_method
from timesigma.algorithms.configuration import (
    resolve_configuration,
    validate_parameter_type as _require_replacement_type,
)
from timesigma.benchmark.leaderboard import OFFICIAL_METRIC_IDS

if TYPE_CHECKING:
    from .preprocessing import PreprocessingSpec


InputState = Literal["raw_train"]
ActionDimension = Literal[
    "algorithm",
    "preprocess",
    "diff_order",
    "window",
    "hyperparameter",
    "preprocessing",
    "ensemble",
    "pipeline",
]

ACTION_DIMENSIONS = (
    "algorithm",
    "preprocess",
    "diff_order",
    "window",
    "hyperparameter",
    "preprocessing",
    "ensemble",
    "pipeline",
)
PREPROCESS_OPTIONS = ("raw", "min-max", "z-score")


@dataclass(frozen=True)
class PlanningUnit:
    curve: str | None
    training_curves: tuple[str, ...]
    held_out_curves: tuple[str, ...]

    def __post_init__(self) -> None:
        training = _curve_names(self.training_curves, "training_curves")
        held_out = _curve_names(
            self.held_out_curves,
            "held_out_curves",
            allow_empty=True,
        )
        if set(training) & set(held_out):
            raise ValueError("planning unit training and held-out curves must be disjoint")
        if self.curve is not None and not self.curve.strip():
            raise ValueError("planning unit curve cannot be empty")
        object.__setattr__(self, "training_curves", training)
        object.__setattr__(self, "held_out_curves", held_out)

    @classmethod
    def for_schema(
        cls,
        schema: str,
        *,
        curve: str | None,
        training_curves: Sequence[str],
        held_out_curves: Sequence[str],
    ) -> PlanningUnit:
        if schema not in EASYTSAD_SCHEMAS:
            raise ValueError(f"unsupported EasyTSAD schema: {schema}")
        unit = cls(curve, tuple(training_curves), tuple(held_out_curves))
        if schema == "naive":
            if unit.curve is None or unit.training_curves != (unit.curve,):
                raise ValueError(
                    "naive planning requires one curve as its sole training member"
                )
            if unit.held_out_curves:
                raise ValueError("naive planning does not accept held-out curves")
        elif schema == "all_in_one":
            if unit.curve is not None:
                raise ValueError("all_in_one planning requires dataset scope")
            if unit.held_out_curves:
                raise ValueError("all_in_one planning does not accept held-out curves")
        elif unit.curve is not None:
            raise ValueError("zero_shot planning requires dataset scope")
        elif not unit.held_out_curves:
            raise ValueError("zero_shot planning requires held-out curves")
        return unit

    @classmethod
    def from_dict(
        cls, schema: str, payload: Mapping[str, Any]
    ) -> PlanningUnit:
        curve = payload.get("curve")
        if curve is not None and not isinstance(curve, str):
            raise TypeError("planning unit curve must be a string or null")
        return cls.for_schema(
            schema,
            curve=curve,
            training_curves=_curve_name_list(payload, "training_curves"),
            held_out_curves=_curve_name_list(payload, "held_out_curves"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "curve": self.curve,
            "training_curves": list(self.training_curves),
            "held_out_curves": list(self.held_out_curves),
        }


@dataclass(frozen=True)
class TaskContext:
    dataset: str
    schema: str
    target_metric: str
    unit: PlanningUnit
    input_state: InputState = "raw_train"

    def __post_init__(self) -> None:
        if not self.dataset.strip():
            raise ValueError("dataset is required")
        if self.schema not in EASYTSAD_SCHEMAS:
            raise ValueError(f"unsupported EasyTSAD schema: {self.schema}")
        if self.target_metric not in OFFICIAL_METRIC_IDS:
            raise ValueError(
                f"unsupported EasyTSAD leaderboard metric: {self.target_metric}"
            )
        if self.input_state != "raw_train":
            raise ValueError("the first Pipeline design space requires raw_train input")
        expected_unit = PlanningUnit.for_schema(
            self.schema,
            curve=self.unit.curve,
            training_curves=self.unit.training_curves,
            held_out_curves=self.unit.held_out_curves,
        )
        if self.unit != expected_unit:
            raise ValueError("planning unit does not match the task schema")

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "schema": self.schema,
            "target_metric": self.target_metric,
            "input_state": self.input_state,
            "unit": self.unit.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskContext:
        dataset = payload.get("dataset")
        schema = payload.get("schema")
        target_metric = payload.get("target_metric")
        input_state = payload.get("input_state", "raw_train")
        if not all(
            isinstance(value, str)
            for value in (dataset, schema, target_metric, input_state)
        ):
            raise TypeError("task context fields must be strings")
        unit = payload.get("unit")
        if not isinstance(unit, Mapping):
            raise TypeError("task context unit must be a mapping")
        return cls(
            dataset=dataset,
            schema=schema,
            target_metric=target_metric,
            unit=PlanningUnit.from_dict(schema, unit),
            input_state=input_state,
        )


@dataclass(frozen=True)
class PipelineSpec:
    method: str
    resolved_configuration: Mapping[str, Any]

    def __post_init__(self) -> None:
        _method_spec(self.method)
        configuration = dict(self.resolved_configuration)
        if set(configuration) != {"Data_Params", "Model_Params"}:
            raise ValueError(
                "resolved configuration requires Data_Params and Model_Params"
            )
        data_params = configuration["Data_Params"]
        model_params = configuration["Model_Params"]
        if not isinstance(data_params, Mapping) or not isinstance(
            model_params, Mapping
        ):
            raise TypeError("resolved configuration sections must be mappings")
        if set(("preprocess", "diff_order")) - set(data_params):
            raise ValueError(
                "Data_Params requires explicit preprocess and diff_order"
            )
        if data_params["preprocess"] not in PREPROCESS_OPTIONS:
            raise ValueError(
                f"unsupported preprocess: {data_params['preprocess']}"
            )
        diff_order = data_params["diff_order"]
        if isinstance(diff_order, bool) or not isinstance(diff_order, int):
            raise TypeError("diff_order must be an integer")
        if diff_order < 0:
            raise ValueError("diff_order must be non-negative")
        if self.method == "IsolationForest":
            from timesigma.algorithms.isolation_forest.binding import validate_data_configuration

            validate_data_configuration(configuration)
        object.__setattr__(
            self,
            "resolved_configuration",
            _freeze_mapping(configuration),
        )

    @classmethod
    def official_default(cls, method: str, dataset: str) -> PipelineSpec:
        spec = _method_spec(method)
        configuration = resolve_configuration(spec.module, (dataset,))[dataset]
        return cls(method=method, resolved_configuration=configuration)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "resolved_configuration": _thaw(self.resolved_configuration),
        }

    def normalized(self) -> NormalizedPipeline:
        return NormalizedPipeline(
            method=self.method,
            resolved_configuration=self.resolved_configuration,
        )

    def identity_key(self) -> str:
        return self.normalized().identity_key()


@dataclass(frozen=True)
class NormalizedPipeline:
    method: str
    resolved_configuration: Mapping[str, Any]
    operators: Sequence[Mapping[str, Any]] = ()
    ensemble_ref: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        spec = PipelineSpec(
            method=self.method,
            resolved_configuration=self.resolved_configuration,
        )
        if isinstance(self.operators, (str, bytes)) or not isinstance(
            self.operators,
            Sequence,
        ):
            raise TypeError("operators must be a sequence of mappings")
        if not all(isinstance(item, Mapping) for item in self.operators):
            raise TypeError("operators must contain mappings")
        operators = tuple(_freeze_mapping(dict(item)) for item in self.operators)
        if self.method == "IsolationForest" and operators:
            raise ValueError(f"{self.method} is frozen without preprocessing components")
        ensemble_ref = (
            None
            if self.ensemble_ref is None
            else _freeze_mapping(dict(self.ensemble_ref))
        )
        object.__setattr__(self, "method", spec.method)
        object.__setattr__(
            self,
            "resolved_configuration",
            spec.resolved_configuration,
        )
        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "ensemble_ref", ensemble_ref)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NormalizedPipeline:
        method = payload.get("method")
        configuration = payload.get("resolved_configuration")
        if not isinstance(method, str) or not isinstance(configuration, Mapping):
            raise TypeError("normalized Pipeline requires method and resolved_configuration")
        pipeline = cls(
            method=method,
            resolved_configuration=configuration,
            operators=_operator_payloads(payload.get("operators", ())),
            ensemble_ref=_optional_mapping(payload.get("ensemble_ref")),
        )
        recorded_identity = payload.get("normalized_pipeline_id")
        if recorded_identity is not None:
            if not isinstance(recorded_identity, str):
                raise TypeError("normalized_pipeline_id must be a string")
            if recorded_identity != pipeline.identity_key():
                raise ValueError("normalized_pipeline_id does not match Pipeline content")
        return pipeline

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "resolved_configuration": _thaw(self.resolved_configuration),
            "operators": [_thaw(item) for item in self.operators],
            "ensemble_ref": (
                None if self.ensemble_ref is None else _thaw(self.ensemble_ref)
            ),
        }

    def preprocessing_spec(self, schema: str) -> PreprocessingSpec:
        from .preprocessing import preprocessing_for_pipeline

        return preprocessing_for_pipeline(self, schema)

    def canonical_view(self, schema: str) -> dict[str, Any]:
        """Category representation; the execution identity remains legacy-compatible."""
        ensemble = {"method": "none", "detectors": []}
        if self.ensemble_ref is not None:
            reference = _thaw(self.ensemble_ref)
            if any(item["resolved_configuration"]["Data_Params"] != _thaw(self.resolved_configuration["Data_Params"]) for item in reference["detectors"]):
                raise ValueError("canonical ensemble requires shared preprocessing")
            ensemble = {**reference, "detectors": [
                {"method": item["method"], "parameters": item["resolved_configuration"]["Model_Params"]}
                for item in reference["detectors"]
            ]}
        return {
            "preprocessing": self.preprocessing_spec(schema).as_dict(),
            "detector": {"method": self.method, "parameters": _thaw(self.resolved_configuration["Model_Params"])},
            "ensemble": ensemble,
        }

    @classmethod
    def from_canonical_view(cls, payload: Mapping[str, Any], schema: str) -> NormalizedPipeline:
        from .preprocessing import PreprocessingSpec

        if set(payload) != {"preprocessing", "detector", "ensemble"}:
            raise ValueError("canonical Pipeline requires preprocessing, detector and ensemble")
        spec = PreprocessingSpec.from_dict(payload["preprocessing"], schema, allow_legacy=True)
        if spec.legacy is not None and spec.legacy["execution_layer"] != "native_pipeline":
            raise ValueError("materialized preprocessing must replay through PipelineDefinition")
        data, operators = spec.to_legacy()
        detector = payload["detector"]
        if set(detector) != {"method", "parameters"}:
            raise ValueError("canonical detector requires method and parameters")
        ensemble = payload["ensemble"]
        reference = None
        if ensemble["method"] == "none":
            if set(ensemble) != {"method", "detectors"} or ensemble["detectors"]:
                raise ValueError("ensemble none cannot contain additional detectors")
        else:
            if set(ensemble) != {"method", "detectors", "normalization", "alignment"}:
                raise ValueError("canonical ensemble requires its frozen fusion contract")
            reference = {**_thaw(ensemble), "detectors": [
                {"method": item["method"], "resolved_configuration": {"Data_Params": data, "Model_Params": _thaw(item["parameters"])}}
                for item in ensemble["detectors"]
            ]}
        result = cls(detector["method"], {"Data_Params": data, "Model_Params": _thaw(detector["parameters"])}, operators, reference)
        if result.canonical_view(schema) != _thaw(payload):
            raise ValueError("canonical Pipeline is not a lossless execution representation")
        return result

    def identity_key(self) -> str:
        return json.dumps(
            self.canonical_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def normalized(self) -> NormalizedPipeline:
        return self

    def as_dict(self) -> dict[str, Any]:
        payload = self.canonical_payload()
        payload["normalized_pipeline_id"] = self.identity_key()
        return payload


@dataclass(frozen=True)
class NextAction:
    dimension: ActionDimension
    value: Any = None
    parameter: str | None = None

    def __post_init__(self) -> None:
        if self.dimension not in ACTION_DIMENSIONS:
            if self.dimension in {"scoring", "postprocess", "post-processing"}:
                raise ValueError(
                    f"{self.dimension} is outside the first Pipeline design space"
                )
            raise ValueError(f"unsupported next action dimension: {self.dimension}")
        if self.value is None:
            raise ValueError(f"{self.dimension} requires a value")
        if self.dimension == "hyperparameter":
            if self.parameter is None or not self.parameter.strip():
                raise ValueError("hyperparameter requires a parameter name")
        elif self.parameter is not None:
            raise ValueError(f"{self.dimension} does not accept a parameter name")
        object.__setattr__(self, "value", _freeze_value(self.value))

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "parameter": self.parameter,
            "value": _thaw(self.value),
        }


def resolve_baseline_pipeline(baseline: Any, dataset: str) -> PipelineSpec | NormalizedPipeline:
    if isinstance(baseline, (PipelineSpec, NormalizedPipeline)):
        return baseline

    candidate = _field(baseline, "candidate_pipeline")
    if candidate is not None:
        return resolve_baseline_pipeline(candidate, dataset)
    nested = _field(baseline, "pipeline")
    if nested is not None:
        return resolve_baseline_pipeline(nested, dataset)

    method = _field(baseline, "method")
    configuration = _field(baseline, "resolved_configuration")
    if configuration is None:
        configuration = _field(baseline, "configuration")
    if not isinstance(method, str) or not isinstance(configuration, Mapping):
        raise TypeError(
            "baseline requires method and resolved_configuration fields"
        )
    if "Data_Params" not in configuration and dataset in configuration:
        configuration = configuration[dataset]
    if not isinstance(configuration, Mapping):
        raise TypeError("baseline resolved_configuration must be a mapping")
    if _field(baseline, "operators") or _field(baseline, "ensemble_ref"):
        return NormalizedPipeline(method, configuration, _field(baseline, "operators") or (), _field(baseline, "ensemble_ref"))
    return PipelineSpec(method, configuration)


def normalize_pipeline(pipeline: Any, dataset: str) -> NormalizedPipeline:
    return resolve_baseline_pipeline(pipeline, dataset).normalized()


def apply_next_action(
    baseline: Any,
    action: NextAction,
    task: TaskContext,
) -> PipelineSpec | NormalizedPipeline:
    from .semantics import context_window_parameter, replace_detector

    current = resolve_baseline_pipeline(baseline, task.dataset)
    if action.dimension in {"preprocessing", "ensemble", "pipeline"}:
        from .planning import normalize_operators, plan_pipeline, validate_planned_pipeline

        if action.dimension == "pipeline":
            return plan_pipeline(action.value, task)
        payload = current.normalized().canonical_payload()
        if action.dimension == "preprocessing":
            payload["operators"] = normalize_operators(action.value)
        else:
            payload["ensemble_ref"] = None if dict(action.value) == {"method": "none"} else _thaw(action.value)
        candidate = NormalizedPipeline.from_dict(payload)
        validate_planned_pipeline(candidate, task)
        return candidate
    if isinstance(current, NormalizedPipeline) and (current.operators or current.ensemble_ref):
        if current.ensemble_ref is not None:
            raise ValueError("modify a standalone detector before constructing a new ensemble")
        candidate = apply_next_action(PipelineSpec(current.method, current.resolved_configuration), action, task)
        return NormalizedPipeline(candidate.method, candidate.resolved_configuration, current.operators)
    current_spec = _method_spec(current.method)
    _require_schema(current.method, current_spec, task.schema)

    if action.dimension == "algorithm":
        if not isinstance(action.value, str):
            raise TypeError("algorithm action requires a method name")
        if action.value == current.method:
            raise ValueError("algorithm action must select a different method")
        next_spec = _method_spec(action.value)
        _require_schema(action.value, next_spec, task.schema)
        return replace_detector(PipelineSpec(current.method, current.resolved_configuration), action.value, task)

    configuration = current.as_dict()["resolved_configuration"]
    data_params = configuration["Data_Params"]
    model_params = configuration["Model_Params"]

    if action.dimension == "preprocess":
        if action.value not in PREPROCESS_OPTIONS:
            raise ValueError(f"unsupported preprocess: {action.value}")
        _replace_value(data_params, "preprocess", action.value, action.dimension)
    elif action.dimension == "diff_order":
        if isinstance(action.value, bool) or not isinstance(action.value, int):
            raise TypeError("diff_order action requires an integer")
        if action.value < 0:
            raise ValueError("diff_order must be non-negative")
        _replace_value(data_params, "diff_order", action.value, action.dimension)
    elif action.dimension == "window":
        parameter = context_window_parameter(current.method)
        if parameter is None:
            raise ValueError(
                f"{current.method} has no Catalog-declared window parameter"
            )
        if isinstance(action.value, bool) or not isinstance(action.value, int):
            raise TypeError("window action requires an integer")
        if action.value <= 0:
            raise ValueError("window must be positive")
        if parameter not in model_params:
            raise ValueError(
                f"resolved configuration is missing Catalog window: {parameter}"
            )
        _replace_value(model_params, parameter, action.value, action.dimension)
    elif action.dimension == "hyperparameter":
        assert action.parameter is not None
        if action.parameter == context_window_parameter(current.method):
            raise ValueError(
                f"{action.parameter} must use the window action dimension"
            )
        if action.parameter not in model_params:
            raise ValueError(
                f"unknown resolved Model_Params parameter: {action.parameter}"
            )
        _replace_value(
            model_params,
            action.parameter,
            _thaw(action.value),
            action.dimension,
        )
    else:
        raise AssertionError(f"unhandled action dimension: {action.dimension}")

    if current_spec.searchable_parameters is not None:
        defaults = PipelineSpec.official_default(current.method, task.dataset).resolved_configuration["Model_Params"]
        current_spec.validate_parameters(model_params, defaults)
    return PipelineSpec(current.method, configuration)


def candidate_pipeline_from_action(
    baseline: Any,
    action: NextAction,
    task: TaskContext,
) -> PipelineSpec | NormalizedPipeline:
    candidate = apply_next_action(baseline, action, task)
    validate_declared_pipeline_change(baseline, action, candidate, task)
    return candidate


def actual_changed_dimension(
    baseline: Any,
    candidate: Any,
    task: TaskContext,
) -> ActionDimension:
    return actual_pipeline_change(baseline, candidate, task)[0]


def actual_pipeline_change(
    baseline: Any,
    candidate: Any,
    task: TaskContext,
) -> tuple[ActionDimension, str | None]:
    return _actual_change(baseline, candidate, task)


def validate_declared_pipeline_change(
    baseline: Any,
    action: NextAction,
    candidate: Any,
    task: TaskContext,
) -> ActionDimension:
    from .semantics import controlled_action

    inferred = controlled_action(baseline, candidate, task)
    if action.dimension != "pipeline" and (inferred.dimension, inferred.parameter) != (action.dimension, action.parameter):
        raise ValueError("actual_changed_dimension does not match the declared action dimension")
    expected = apply_next_action(baseline, action, task)
    if expected.normalized() != normalize_pipeline(candidate, task.dataset):
        raise ValueError("action does not match the complete Candidate Pipeline")
    return inferred.dimension


def _actual_change(
    baseline: Any,
    candidate: Any,
    task: TaskContext,
) -> tuple[ActionDimension, str | None]:
    from .semantics import context_window_parameter, controlled_action

    action = controlled_action(baseline, candidate, task)
    parameter = context_window_parameter(normalize_pipeline(candidate, task.dataset).method) if action.dimension == "window" else action.parameter
    return action.dimension, parameter


def _replace_value(
    parameters: dict[str, Any], key: str, value: Any, dimension: str
) -> None:
    _require_replacement_type(parameters[key], value, key)
    if parameters[key] == value:
        raise ValueError(f"{dimension} action must change its baseline value")
    parameters[key] = deepcopy(value)


def _method_spec(method: str) -> AlgorithmSpec:
    try:
        return get_method(method)
    except KeyError as exc:
        raise ValueError(f"unknown Catalog method: {method}") from exc


def _require_schema(method: str, spec: AlgorithmSpec, schema: str) -> None:
    if not spec.agent_available:
        raise ValueError(f"{method} is not available to Agent planning until qualification completes")
    capability = spec.capability(schema)
    if not capability.supported:
        raise ValueError(
            f"{method} is N/A for {schema}: {capability.unavailable_reason}"
        )


def _field(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _curve_names(
    values: Sequence[str], label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"planning unit {label} must be a sequence of strings")
    names = tuple(values)
    if not allow_empty and not names:
        raise ValueError(f"planning unit {label} cannot be empty")
    if not all(isinstance(name, str) and name.strip() for name in names):
        raise TypeError(f"planning unit {label} must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"planning unit {label} must be unique")
    return names


def _curve_name_list(
    payload: Mapping[str, Any], label: str
) -> tuple[str, ...]:
    values = payload.get(label)
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"planning unit {label} must be a list")
    return tuple(values)


def _operator_payloads(value: Any) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("operators must be a sequence")
    if not all(isinstance(item, Mapping) for item in value):
        raise TypeError("operators must contain mappings")
    return tuple(value)


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("ensemble_ref must be a mapping or null")
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not all(isinstance(key, str) for key in value):
        raise TypeError("configuration keys must be strings")
    return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return deepcopy(value)


__all__ = [
    "ACTION_DIMENSIONS",
    "ActionDimension",
    "NextAction",
    "NormalizedPipeline",
    "PlanningUnit",
    "PipelineSpec",
    "TaskContext",
    "actual_changed_dimension",
    "actual_pipeline_change",
    "apply_next_action",
    "candidate_pipeline_from_action",
    "normalize_pipeline",
    "resolve_baseline_pipeline",
    "validate_declared_pipeline_change",
]
