from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from timesigma.algorithms import get_method

from .pipeline import NextAction, NormalizedPipeline, PipelineSpec, TaskContext, normalize_pipeline


def context_window_parameter(method: str) -> str | None:
    # MatrixProfile has a subsequence window but no training-sample lifecycle.
    return "window" if method == "MatrixProfile" else get_method(method).sample_spec.window_parameter


def execution_only_parameters() -> frozenset[str]:
    return frozenset({"n_jobs", "verbose", "cache_size"})


def window_contract(method: str) -> dict[str, Any]:
    return {
        "parameter": context_window_parameter(method),
        "fixed_value": 1 if context_window_parameter(method) is None else None,
        "minimum": 4 if method == "MatrixProfile" else 1,
        "meaning": "Number of observations per input/subsequence window; scalar detectors use one observation.",
        "horizon": get_method(method).sample_spec.horizon,
        "note": (
            "FITS context is the out_len raw-observation reconstruction window, internally decimated to fixed input_len=25; out_len must be even and divisible by input_len. cut_freq must fit both FFT supports. Score suffix starts at out_len-1."
            if method == "FITS" else
            "DCdetector win_size must be divisible by frozen patches [1,3,5] and <=5000. Training windows slide at step=1; test uses non-overlapping blocks covering the suffix starting at N % win_size. d_model must be even."
            if method == "DCdetector" else
            "Equal window length does not imply equal forecasting/reconstruction semantics or native score coverage."
        ),
    }


def _window(pipeline: NormalizedPipeline | PipelineSpec) -> Any:
    parameter = context_window_parameter(pipeline.method)
    return pipeline.resolved_configuration["Model_Params"][parameter] if parameter else 1


def _stride(pipeline: NormalizedPipeline | PipelineSpec) -> Any:
    parameter = get_method(pipeline.method).sample_spec.step_parameter
    return pipeline.resolved_configuration["Model_Params"][parameter] if parameter else 1


def validate_detector_context(pipeline: NormalizedPipeline | PipelineSpec) -> None:
    contract = window_contract(pipeline.method)
    value = _window(pipeline)
    if isinstance(value, bool) or not isinstance(value, int) or value < contract["minimum"]:
        raise ValueError(f"{pipeline.method} context window must be an integer >= {contract['minimum']}")
    stride = _stride(pipeline)
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("sampling stride must be a positive integer")
    if stride != 1:
        raise ValueError("Native evaluation requires sampling stride=1 for continuous suffix score alignment; no strided-score adapter is implemented")
    if pipeline.method == "AnomalyTransformer":
        parameters = pipeline.resolved_configuration["Model_Params"]
        if parameters["input_c"] != 1 or parameters["output_c"] != 1:
            raise ValueError("Univariate Schema requires AnomalyTransformer input_c=output_c=1")


def replace_detector(baseline: PipelineSpec, method: str, task: TaskContext) -> PipelineSpec:
    target = get_method(method)
    if not target.capability(task.schema).supported:
        raise ValueError(f"{method} is N/A for {task.schema}")
    configuration = PipelineSpec.official_default(method, task.dataset).as_dict()["resolved_configuration"]
    configuration["Data_Params"] = deepcopy(baseline.as_dict()["resolved_configuration"]["Data_Params"])
    for name, value, parameter in (
        ("context window", _window(baseline), context_window_parameter(method)),
        ("sampling stride", _stride(baseline), target.sample_spec.step_parameter),
    ):
        if parameter is None:
            if value != 1:
                raise ValueError(f"Cannot preserve {name}={value}: {method} supports only {name}=1; choose another baseline or research action")
        else:
            configuration["Model_Params"][parameter] = value
    return PipelineSpec(method, configuration)


def semantic_pipeline_diff(baseline: Any, candidate: Any, task: TaskContext) -> dict[str, Any]:
    before = normalize_pipeline(baseline, task.dataset)
    after = normalize_pipeline(candidate, task.dataset)
    changes: list[dict[str, Any]] = []
    initialization = []

    def change(dimension: str, semantic: str, old: Any, new: Any, *, parameter: str | None = None, branch: str = "primary") -> None:
        if json.dumps(old, sort_keys=True) != json.dumps(new, sort_keys=True):
            changes.append({"dimension": dimension, "semantic_dimension": semantic, "branch": branch, "parameter": parameter, "before": old, "after": new})

    def detector_diff(old: NormalizedPipeline | PipelineSpec, new: NormalizedPipeline | PipelineSpec, branch: str) -> None:
        old_config = old.normalized().canonical_payload()["resolved_configuration"]
        new_config = new.normalized().canonical_payload()["resolved_configuration"]
        for key in sorted(set(old_config["Data_Params"]) | set(new_config["Data_Params"])):
            dimension, semantic = {"preprocess": ("preprocess", "native_scaling"), "diff_order": ("diff_order", "differencing")}.get(key, ("unsupported", "data_config:" + key))
            change(dimension, semantic, old_config["Data_Params"].get(key, {"absent": True}), new_config["Data_Params"].get(key, {"absent": True}), branch=branch)
        change("window", "context_window", _window(old), _window(new), parameter=context_window_parameter(new.method), branch=branch)
        change("hyperparameter", "sampling_stride", _stride(old), _stride(new), parameter=get_method(new.method).sample_spec.step_parameter, branch=branch)
        change("algorithm", "detector_identity", old.method, new.method, branch=branch)
        shared = {context_window_parameter(new.method), get_method(new.method).sample_spec.step_parameter}
        old_params = old_config["Model_Params"]
        if old.method != new.method:
            old_params = PipelineSpec.official_default(new.method, task.dataset).as_dict()["resolved_configuration"]["Model_Params"]
            initialization.append({
                "branch": branch, "method": new.method, "policy": "target_method_private_defaults",
                "parameters": {k: v for k, v in old_params.items() if k not in shared},
                "note": "Declared detector-private initialization, not an independent research finding; shared window/stride and Data_Params are preserved separately.",
            })
        for key in sorted((set(old_params) | set(new_config["Model_Params"])) - shared):
            execution_only = key in execution_only_parameters()
            change("unsupported" if execution_only else "hyperparameter", ("execution_parameter:" if execution_only else "private_parameter:") + key, old_params.get(key, {"absent": True}), new_config["Model_Params"].get(key, {"absent": True}), parameter=key, branch=branch)

    detector_diff(before, after, "primary")
    old_payload, new_payload = before.canonical_payload(), after.canonical_payload()
    old_ops = {op["component_id"]: op for op in old_payload["operators"]}
    new_ops = {op["component_id"]: op for op in new_payload["operators"]}
    for component in sorted(set(old_ops) | set(new_ops)):
        change("preprocessing", "component:" + component, old_ops.get(component), new_ops.get(component))
    old_order = [name for name in old_ops if name in new_ops]
    new_order = [name for name in new_ops if name in old_ops]
    change("preprocessing", "preprocessing_order", old_order, new_order)
    old_fusion, new_fusion = old_payload["ensemble_ref"], new_payload["ensemble_ref"]
    if old_fusion is None or new_fusion is None:
        change("ensemble", "composition", old_fusion, new_fusion)
    else:
        old_branches, new_branches = old_fusion["detectors"], new_fusion["detectors"]
        for index, (old, new) in enumerate(zip(old_branches, new_branches)):
            detector_diff(PipelineSpec(**old), PipelineSpec(**new), f"ensemble.detectors[{index}]")
        change("ensemble", "composition", {k: v for k, v in old_fusion.items() if k != "detectors"}, {k: v for k, v in new_fusion.items() if k != "detectors"})
        change("ensemble", "branch_count", len(old_branches), len(new_branches))
    return {"changes": changes, "single_semantic_change": len(changes) == 1 and changes[0]["dimension"] != "unsupported", "detector_private_initialization": initialization}


def controlled_action(baseline: Any, candidate: Any, task: TaskContext) -> NextAction:
    from .planning import validate_planned_pipeline

    before, after = normalize_pipeline(baseline, task.dataset), normalize_pipeline(candidate, task.dataset)
    validate_planned_pipeline(before, task)
    validate_planned_pipeline(after, task)
    diff = semantic_pipeline_diff(before, after, task)
    changes = diff["changes"]
    if not changes:
        raise ValueError("Candidate Pipeline must differ from the baseline Pipeline (no-op)")
    if len(changes) != 1:
        dimensions = ", ".join(item["semantic_dimension"] for item in changes)
        raise ValueError("Candidate Pipeline must change exactly one semantic research dimension; actual changes: " + dimensions)
    changed = changes[0]
    if changed["dimension"] == "unsupported":
        raise ValueError("Unsupported research change (execution settings and extra Data_Params are not research actions): " + changed["semantic_dimension"])
    if before.ensemble_ref is not None and after.ensemble_ref is not None and (
        before.method != after.method or before.resolved_configuration != after.resolved_configuration or before.ensemble_ref != after.ensemble_ref
    ):
        raise ValueError("Modify a standalone detector before constructing a new ensemble; existing ensemble branches must stay frozen")
    if changed["branch"] != "primary":
        raise ValueError("Existing ensemble branch changes require a standalone research baseline")
    dimension = changed["dimension"]
    value = changed["after"]
    if dimension == "preprocessing":
        value = after.canonical_payload()["operators"]
    elif dimension == "ensemble":
        value = after.canonical_payload()["ensemble_ref"] or {"method": "none"}
    return NextAction(dimension, value, changed["parameter"] if dimension == "hyperparameter" else None)
