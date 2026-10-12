from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from timesigma.algorithms import get_method

from .pipeline import NextAction, NormalizedPipeline, PipelineSpec, TaskContext


def plan_pipeline(payload: Mapping[str, Any], task: TaskContext) -> NormalizedPipeline:
    """Resolve a complete proposal using native Method defaults, never test feedback."""
    _fields(payload, {"preprocessing", "detector", "ensemble"})
    preprocessing = _mapping(payload["preprocessing"])
    from .preprocessing import PreprocessingSpec

    spec = (PreprocessingSpec.from_flat_proposal(preprocessing, task.schema)
            if "differencing" in preprocessing else PreprocessingSpec.from_dict(preprocessing, task.schema))
    data_params, operators = spec.to_legacy()
    primary = _detector(_mapping(payload["detector"]), task, data_params)
    ensemble = _mapping(payload["ensemble"])
    _fields(ensemble, {"method", "detectors"})
    detectors = ensemble["detectors"]
    if not isinstance(detectors, (list, tuple)):
        raise TypeError("ensemble.detectors must be an array")
    reference = None
    if ensemble["method"] == "none":
        if detectors:
            raise ValueError("ensemble none cannot contain additional detectors")
    elif ensemble["method"] == "mean":
        if len(detectors) != 1:
            raise ValueError("mean fusion requires exactly one additional detector")
        secondary = _detector(_mapping(detectors[0]), task, data_params)
        if secondary == primary:
            raise ValueError("mean fusion requires different detector configurations")
        reference = {
            "method": "mean",
            "detectors": [secondary.as_dict()],
            "normalization": "average_rank",
            "alignment": "equal_native_suffix",
        }
    else:
        raise ValueError("unsupported ensemble method")
    planned = NormalizedPipeline(
        primary.method, primary.resolved_configuration, operators, reference,
    )
    validate_planned_pipeline(planned, task)
    return planned


def normalize_operators(operators: Any) -> list[dict[str, Any]]:
    if not isinstance(operators, (list, tuple)):
        raise TypeError("operators must be an array")
    if len(operators) > 2:
        raise ValueError("Pipeline supports at most two preprocessing components")
    resolved = []
    for raw in operators:
        component = _mapping(raw)
        if set(component) - {"component_id", "parameters"}:
            raise ValueError("unknown preprocessing component fields")
        name = component["component_id"]
        parameters = dict(_mapping(component.get("parameters", {})))
        if name == "robust_scaler":
            _fields(parameters, set())
        elif name == "moving_average":
            if set(parameters) - {"window"}:
                raise ValueError("unknown moving_average parameter")
            window = parameters.setdefault("window", 5)
            if isinstance(window, bool) or not isinstance(window, int) or window <= 1:
                raise ValueError("moving_average window must be an integer greater than one")
        else:
            raise ValueError(f"unsupported preprocessing component: {name}")
        if any(item["component_id"] == name for item in resolved):
            raise ValueError("duplicate preprocessing component")
        resolved.append({"component_id": name, "parameters": parameters})
    return resolved


def validate_planned_pipeline(
    pipeline: NormalizedPipeline, task: TaskContext,
) -> None:
    from .semantics import validate_detector_context

    normalize_operators(pipeline.canonical_payload()["operators"])
    for detector in pipeline_detectors(pipeline):
        validate_detector_context(detector)
        spec = get_method(detector.method)
        if not spec.agent_available:
            raise ValueError(f"{detector.method} is not available to Agent planning until qualification completes")
        if not spec.capability(task.schema).supported:
            raise ValueError(f"{detector.method} is N/A for {task.schema}")
        defaults = PipelineSpec.official_default(detector.method, task.dataset).resolved_configuration["Model_Params"]
        if set(detector.resolved_configuration["Model_Params"]) != set(defaults):
            raise ValueError("Resolved detector parameters must match the declared Method parameter names")
        spec.validate_parameters(detector.resolved_configuration["Model_Params"], defaults)
    if pipeline.ensemble_ref is not None:
        reference = pipeline.canonical_payload()["ensemble_ref"]
        _fields(reference, {"method", "detectors", "normalization", "alignment"})
        contract = (reference["method"], reference["normalization"], reference["alignment"])
        if contract != ("mean", "average_rank", "equal_native_suffix"):
            raise ValueError("unsupported mean fusion contract")
        detectors = pipeline_detectors(pipeline)
        if len(detectors) != 2 or detectors[0] == detectors[1]:
            raise ValueError("mean fusion requires two different detector configurations")
        if (
            detectors[0].resolved_configuration["Data_Params"]
            != detectors[1].resolved_configuration["Data_Params"]
        ):
            raise ValueError("mean fusion uses shared preprocessing")
        if any(detector.method not in {"LOF", "OCSVM", "AR", "AE"} for detector in detectors):
            raise ValueError(
                "mean fusion supports the audited LOF, OCSVM, AR and AE score alignment contracts"
            )
        offsets = []
        for detector in detectors:
            sample = get_method(detector.method).sample_spec
            window = (
                detector.resolved_configuration["Model_Params"][sample.window_parameter]
                if sample.window_parameter is not None else 1
            )
            if isinstance(window, bool) or not isinstance(window, int) or window < 1:
                raise ValueError("detector window must be a positive integer")
            offsets.append(window + sample.horizon - 1)
        if offsets[0] != offsets[1]:
            raise ValueError("mean fusion requires identical native score support")


def pipeline_detectors(pipeline: NormalizedPipeline) -> tuple[PipelineSpec, ...]:
    detectors = [PipelineSpec(pipeline.method, pipeline.resolved_configuration)]
    if pipeline.ensemble_ref is not None:
        reference = pipeline.canonical_payload()["ensemble_ref"]
        others = reference.get("detectors")
        if not isinstance(others, list) or len(others) != 1:
            raise ValueError("mean fusion requires exactly one additional detector")
        other = _mapping(others[0])
        _fields(other, {"method", "resolved_configuration"})
        detectors.append(PipelineSpec(other["method"], _mapping(other["resolved_configuration"])))
    return tuple(detectors)


def infer_planning_action(
    baseline: NormalizedPipeline,
    candidate: NormalizedPipeline,
    task: TaskContext,
) -> NextAction:
    from .semantics import controlled_action

    return controlled_action(baseline, candidate, task)


def _detector(
    payload: Mapping[str, Any],
    task: TaskContext,
    data_params: Mapping[str, Any],
) -> PipelineSpec:
    _fields(payload, {"method", "parameters"})
    method = payload["method"]
    if not isinstance(method, str):
        raise TypeError("detector.method must be a Catalog name")
    default = PipelineSpec.official_default(method, task.dataset)
    raw_parameters = payload["parameters"]
    if isinstance(raw_parameters, (list, tuple)):
        parameters = {}
        for item in raw_parameters:
            item = _mapping(item)
            _fields(item, {"name", "value"})
            if not isinstance(item["name"], str) or item["name"] in parameters:
                raise ValueError("detector parameter names must be unique strings")
            parameters[item["name"]] = item["value"]
    else:
        parameters = _mapping(raw_parameters)
    configuration = default.as_dict()["resolved_configuration"]
    if set(parameters) - set(configuration["Model_Params"]):
        raise ValueError("unknown detector parameter")
    from .pipeline import _require_replacement_type

    for name, value in parameters.items():
        _require_replacement_type(configuration["Model_Params"][name], value, name)
    configuration["Model_Params"].update(deepcopy(dict(parameters)))
    configuration["Data_Params"].update(data_params)
    return PipelineSpec(method, configuration)


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Pipeline section must be an object")
    return value


def _fields(payload: Mapping[str, Any], fields: set[str]) -> None:
    if set(payload) != fields:
        raise ValueError(f"Pipeline fields must be {sorted(fields)}")
