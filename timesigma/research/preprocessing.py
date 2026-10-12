from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from timesigma.algorithms import EASYTSAD_SCHEMAS, get_method


SCALING_METHODS = ("raw", "min_max", "z_score", "robust")
_NATIVE_NAMES = {"raw": "raw", "min-max": "min_max", "z-score": "z_score"}
_EXECUTION_NAMES = {value: key for key, value in _NATIVE_NAMES.items()}


def scaling_scope(method: str, schema: str, *, materialized: bool = False) -> str:
    if schema not in EASYTSAD_SCHEMAS:
        raise ValueError(f"unsupported preprocessing Schema: {schema}")
    if method == "raw":
        return "not_applicable"
    if method == "robust":
        if materialized:
            return "per_curve_full_train_before_split"
        return "per_curve_train" if schema == "naive" else "source_train_members"
    return "per_curve_train"


@dataclass(frozen=True)
class PreprocessingSpec:
    """Three search categories, with an explicit lossless legacy execution layout."""

    scaling: Mapping[str, Any]
    temporal_transform: Mapping[str, Any]
    smoothing: Mapping[str, Any]
    legacy: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.legacy is None:
            _fields(_mapping(self.scaling), {"method", "fit_scope"})
            if self.scaling["method"] not in SCALING_METHODS:
                raise ValueError("scaling must choose exactly one declared method")
            _method_parameter(_mapping(self.temporal_transform), "difference", "order", minimum=1)
            _method_parameter(_mapping(self.smoothing), "moving_average", "window", minimum=2)
        for name in ("scaling", "temporal_transform", "smoothing"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        if self.legacy is not None:
            object.__setattr__(self, "legacy", _freeze(self.legacy))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], schema: str, *, allow_legacy: bool = False) -> PreprocessingSpec:
        required = {"scaling", "temporal_transform", "smoothing"}
        if "legacy" in payload:
            _fields(payload, required | {"legacy"})
            if not allow_legacy:
                raise ValueError("legacy preprocessing layouts are replay-only, not new search proposals")
            legacy = _mapping(payload["legacy"])
            _fields(legacy, {"execution_layer", "data_params", "operators"})
            result = cls.from_legacy(legacy["data_params"], legacy["operators"], schema, execution_layer=legacy["execution_layer"])
            if result.as_dict() != _plain(payload):
                raise ValueError("canonical preprocessing disagrees with its legacy execution layout")
            return result
        _fields(payload, required)
        scaling = _mapping(payload["scaling"])
        _fields(scaling, {"method", "fit_scope"})
        method = scaling["method"]
        if not isinstance(method, str) or method not in SCALING_METHODS:
            raise ValueError("scaling must choose exactly one of raw, min_max, z_score, robust")
        if scaling["fit_scope"] != scaling_scope(method, schema):
            raise ValueError("scaling fit_scope must match the existing method/Schema execution contract")
        temporal = _mapping(payload["temporal_transform"])
        order = _method_parameter(temporal, "difference", "order", minimum=1)
        smoothing = _mapping(payload["smoothing"])
        window = _method_parameter(smoothing, "moving_average", "window", minimum=2)
        return cls(dict(scaling), {"method": "none"} if order is None else {"method": "difference", "order": order},
                   {"method": "none"} if window is None else {"method": "moving_average", "window": window})

    @classmethod
    def from_legacy(cls, data_params: Mapping[str, Any], operators: Sequence[Mapping[str, Any]], schema: str,
                    *, execution_layer: str = "native_pipeline") -> PreprocessingSpec:
        from .planning import normalize_operators

        data = _plain(_mapping(data_params))
        native = _NATIVE_NAMES[data["preprocess"]]
        order = data["diff_order"]
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise ValueError("diff_order must be a non-negative integer")
        materialized = execution_layer == "materialized_before_native"
        if execution_layer not in {"native_pipeline", "materialized_before_native"}:
            raise ValueError("unknown legacy preprocessing execution layer")
        original_ops = _plain(operators)
        if not materialized:
            ops = normalize_operators(operators)
        else:
            ops = []
            for raw in operators:
                name = raw["component_id"]
                parameters = dict(raw.get("parameters", {}))
                if name == "robust_scaler":
                    ops.append({"component_id": name, "parameters": parameters})
                elif name == "moving_average":
                    window = parameters.get("window", 5)
                    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
                        raise ValueError("legacy moving_average window must be positive")
                    ops.append({"component_id": name, "parameters": {**parameters, "window": window}})
                else:
                    raise ValueError(f"unknown materialized preprocessing: {name}")
        scalers = ([{"method": native, "fit_scope": scaling_scope(native, schema)}] if native != "raw" else [])
        robust = [{"method": "robust", "fit_scope": scaling_scope("robust", schema, materialized=materialized)}
                  for op in ops if op["component_id"] == "robust_scaler"]
        scalers = robust + scalers if materialized else scalers + robust
        smoothing = [{"method": "moving_average", "window": op["parameters"]["window"]}
                     for op in ops if op["component_id"] == "moving_average"]
        scaling = _category(scalers, {"method": "raw", "fit_scope": "not_applicable"})
        smooth = _category(smoothing, {"method": "none"})
        # A category view cannot silently reorder MA -> robust, collapse chained
        # scalers (native min-max clips), or move a pre-split materialized fit.
        names = [op["component_id"] for op in ops]
        legacy_needed = materialized or len(scalers) > 1 or names == ["moving_average", "robust_scaler"] or set(data) != {"preprocess", "diff_order"} or original_ops != ops
        legacy = {"execution_layer": execution_layer, "data_params": data, "operators": original_ops} if legacy_needed else None
        return cls(scaling, {"method": "none"} if order == 0 else {"method": "difference", "order": order}, smooth, legacy)

    @classmethod
    def from_flat_proposal(cls, payload: Mapping[str, Any], schema: str) -> PreprocessingSpec:
        """Keep the previous complete-plan API without changing its defaults."""
        _fields(payload, {"differencing", "moving_average", "scaling"})
        method = {"none": "raw", "min-max": "min_max", "z-score": "z_score", "robust": "robust"}.get(payload["scaling"])
        if method is None:
            raise ValueError("unsupported Pipeline scaling")
        order, window = payload["differencing"], payload["moving_average"]
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise ValueError("differencing requires a non-negative integer")
        return cls.from_dict({
            "scaling": {"method": method, "fit_scope": scaling_scope(method, schema)},
            "temporal_transform": {"method": "none"} if order == 0 else {"method": "difference", "order": order},
            "smoothing": {"method": "none"} if window in (None, "none") else {"method": "moving_average", "window": window},
        }, schema)

    def as_dict(self) -> dict[str, Any]:
        result = {name: _plain(getattr(self, name)) for name in ("scaling", "temporal_transform", "smoothing")}
        if self.legacy is not None:
            result["legacy"] = _plain(self.legacy)
        return result

    def to_legacy(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if self.legacy is not None:
            return _plain(self.legacy["data_params"]), _plain(self.legacy["operators"])
        method = self.scaling["method"]
        data = {"preprocess": "raw" if method == "robust" else _EXECUTION_NAMES[method], "diff_order": self.temporal_transform.get("order", 0)}
        ops = [{"component_id": "robust_scaler", "parameters": {}}] if method == "robust" else []
        if self.smoothing["method"] == "moving_average":
            ops.append({"component_id": "moving_average", "parameters": {"window": self.smoothing["window"]}})
        return data, ops

    def execution_order(self) -> list[str]:
        data, ops = self.to_legacy()
        native = (["difference"] if data["diff_order"] else []) + ([data["preprocess"]] if data["preprocess"] != "raw" else [])
        outer = [op["component_id"] for op in ops]
        if self.legacy is not None and self.legacy["execution_layer"] == "materialized_before_native":
            return [*outer, "native_split", *native]
        return ["native_split", *native, *outer]


def preprocessing_for_pipeline(pipeline: Any, schema: str) -> PreprocessingSpec:
    normalized = pipeline.normalized()
    return PreprocessingSpec.from_legacy(normalized.resolved_configuration["Data_Params"], normalized.operators, schema)


def preprocessing_for_definition(definition: Any, schema: str) -> PreprocessingSpec:
    from .experiment import _detector_pipeline_for_execution, _materialized_preprocessing

    detector = _detector_pipeline_for_execution(definition)
    return PreprocessingSpec.from_legacy(detector.resolved_configuration["Data_Params"], _materialized_preprocessing(definition), schema,
                                         execution_layer="materialized_before_native")


def validate_search_preprocessing(baseline: Any, candidate: Any, schema: str) -> None:
    """New Agent proposals do not introduce chains; existing layouts can persist."""
    before, after = preprocessing_for_pipeline(baseline, schema), preprocessing_for_pipeline(candidate, schema)
    if after.scaling["method"] == "legacy_chain" and after.scaling != before.scaling:
        raise ValueError("new search chooses one scaling method; chained scaling is legacy replay-only")


def preprocessing_capability(method: str, schema: str) -> dict[str, Any]:
    spec = get_method(method)
    supported = spec.capability(schema).supported
    if method == "IsolationForest":
        return {
            "supported_scaling": ["raw"] if supported else [],
            "supported_temporal_transform": ["none"] if supported else [],
            "supported_smoothing": ["none"] if supported else [],
            "native_required_preprocessing": "raw, diff_order=0, no components",
            "supports_naive": spec.capability("naive").supported,
            "supports_all_in_one": spec.capability("all_in_one").supported,
            "supports_zero_shot": spec.capability("zero_shot").supported,
            "scaling_fit_scopes": {"raw": "not_applicable"},
            "chained_scaling": {"new_search": False, "legacy_replay": False},
            "basis": "Qualified point-wise raw IsolationForest identity; preprocessing search is not qualified.",
            "defaults_are_requirements": True,
        }
    return {
        "supported_scaling": list(SCALING_METHODS) if supported else [],
        "supported_temporal_transform": ["none", "difference"] if supported else [],
        "supported_smoothing": ["none", "moving_average"] if supported else [],
        "native_required_preprocessing": None,
        "supports_naive": spec.capability("naive").supported,
        "supports_all_in_one": spec.capability("all_in_one").supported,
        "supports_zero_shot": spec.capability("zero_shot").supported,
        "scaling_fit_scopes": {name: scaling_scope(name, schema) for name in SCALING_METHODS},
        "chained_scaling": {"new_search": False, "legacy_replay": True},
        "basis": "Existing univariate, length-preserving input adapter and native Schema lifecycle; does not guarantee feasibility for every dataset/parameter.",
        "defaults_are_requirements": False,
    }


def preprocessing_search_space(schema: str) -> dict[str, Any]:
    return {
        "scaling": {"methods": list(SCALING_METHODS), "fit_scope_by_method": {method: scaling_scope(method, schema) for method in SCALING_METHODS}},
        "temporal_transform": {"methods": ["none", "difference"], "difference": {"order": "integer >= 1"}},
        "smoothing": {"methods": ["none", "moving_average"], "moving_average": {"window": "integer >= 2"}},
        "constraints": ["At most one operation per category; categories can compose", "fit_scope is derived, not tunable", "New search cannot introduce chained scaling; legacy layouts remain replayable", "Category normalization does not merge existing controlled research dimensions"],
    }


def _method_parameter(payload: Mapping[str, Any], method: str, parameter: str, *, minimum: int) -> int | None:
    if payload.get("method") == "none":
        _fields(payload, {"method"})
        return None
    _fields(payload, {"method", parameter})
    if payload["method"] != method:
        raise ValueError(f"only none or {method} is supported")
    value = payload[parameter]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{method}.{parameter} must be an integer >= {minimum}")
    return value


def _category(steps: list[dict[str, Any]], empty: dict[str, Any]) -> dict[str, Any]:
    return empty if not steps else steps[0] if len(steps) == 1 else {"method": "legacy_chain", "stages": steps}


def _fields(payload: Mapping[str, Any], expected: set[str]) -> None:
    if set(payload) != expected:
        raise ValueError(f"preprocessing fields must be {sorted(expected)}")


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("preprocessing category must be an object")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return deepcopy(value)
