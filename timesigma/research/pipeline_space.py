from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from timesigma.algorithms import EASYTSAD_SCHEMAS, METHOD_CATALOG

from .pipeline import PipelineSpec, TaskContext

if TYPE_CHECKING:
    from .preprocessing import PreprocessingSpec


PipelineStage = Literal[
    "preprocessing",
    "representation",
    "detector",
    "postprocessing",
    "composition",
]


@dataclass(frozen=True)
class PipelineComponent:
    component_id: str
    stage: PipelineStage
    input_type: str
    output_type: str
    schemas: tuple[str, ...]
    execution_adapter: str
    parameters: Mapping[str, Any]
    label_free: bool = True

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError("pipeline component_id is required")
        if self.stage not in {
            "preprocessing",
            "representation",
            "detector",
            "postprocessing",
            "composition",
        }:
            raise ValueError(f"unsupported pipeline component stage: {self.stage}")
        if not self.input_type.strip() or not self.output_type.strip():
            raise ValueError("pipeline component input/output types are required")
        if not self.execution_adapter.strip():
            raise ValueError("pipeline component execution_adapter is required")
        if not self.schemas:
            raise ValueError("pipeline component schemas cannot be empty")
        unknown = set(self.schemas) - set(EASYTSAD_SCHEMAS)
        if unknown:
            raise ValueError(f"pipeline component has unsupported schemas: {sorted(unknown)}")
        object.__setattr__(self, "schemas", tuple(self.schemas))
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(deepcopy(dict(self.parameters))),
        )

    def supports(self, schema: str) -> bool:
        return schema in self.schemas

    def as_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "stage": self.stage,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "schemas": list(self.schemas),
            "execution_adapter": self.execution_adapter,
            "parameters": deepcopy(dict(self.parameters)),
            "label_free": self.label_free,
        }


@dataclass(frozen=True)
class PipelineDefinition:
    preprocessing: tuple[Mapping[str, Any], ...]
    representation: tuple[Mapping[str, Any], ...]
    detector: Mapping[str, Any] | None
    postprocessing: tuple[Mapping[str, Any], ...]
    composition: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "preprocessing", _component_refs(self.preprocessing))
        object.__setattr__(self, "representation", _component_refs(self.representation))
        object.__setattr__(self, "detector", _optional_mapping(self.detector))
        object.__setattr__(
            self,
            "postprocessing",
            _component_refs(self.postprocessing),
        )
        object.__setattr__(self, "composition", _optional_mapping(self.composition))

    @classmethod
    def from_pipeline_spec(cls, pipeline: PipelineSpec | Mapping[str, Any]) -> PipelineDefinition:
        spec = (
            pipeline
            if isinstance(pipeline, PipelineSpec)
            else PipelineSpec(
                method=str(pipeline["method"]),
                resolved_configuration=_mapping(pipeline, "resolved_configuration"),
            )
        )
        payload = spec.as_dict()
        configuration = payload["resolved_configuration"]
        data_params = configuration["Data_Params"]
        return cls(
            preprocessing=(
                {
                    "component_id": "native_preprocess",
                    "parameters": {"value": data_params["preprocess"]},
                },
                {
                    "component_id": "native_diff_order",
                    "parameters": {"order": data_params["diff_order"]},
                },
            ),
            representation=(),
            detector={
                "component_id": f"detector:{spec.method}",
                "method": spec.method,
                "resolved_configuration": configuration,
            },
            postprocessing=(),
            composition=None,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PipelineDefinition:
        return cls(
            preprocessing=_sequence(payload.get("preprocessing", ()), "preprocessing"),
            representation=_sequence(payload.get("representation", ()), "representation"),
            detector=_optional_mapping(payload.get("detector")),
            postprocessing=_sequence(payload.get("postprocessing", ()), "postprocessing"),
            composition=_optional_mapping(payload.get("composition")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "preprocessing": [_thaw(item) for item in self.preprocessing],
            "representation": [_thaw(item) for item in self.representation],
            "detector": None if self.detector is None else _thaw(self.detector),
            "postprocessing": [_thaw(item) for item in self.postprocessing],
            "composition": None if self.composition is None else _thaw(self.composition),
        }

    def to_pipeline_spec(self) -> PipelineSpec:
        if self.representation or self.postprocessing or self.composition is not None:
            raise ValueError("only native single-detector PipelineDefinitions convert to PipelineSpec")
        if self.detector is None:
            raise ValueError("single-detector PipelineDefinition requires detector")
        method = _text(self.detector.get("method"), "detector.method")
        configuration = _thaw(_mapping(self.detector, "resolved_configuration"))
        data_params = _thaw(_mapping(configuration, "Data_Params"))
        for component in self.preprocessing:
            component_id = _text(component.get("component_id"), "component_id")
            parameters = _mapping(component, "parameters")
            if component_id == "native_preprocess":
                data_params["preprocess"] = _text(parameters.get("value"), "preprocess")
            elif component_id == "native_diff_order":
                order = parameters.get("order")
                if isinstance(order, bool) or not isinstance(order, int):
                    raise TypeError("native diff_order must be an integer")
                data_params["diff_order"] = order
            else:
                raise ValueError(f"cannot convert non-native preprocessing: {component_id}")
        configuration["Data_Params"] = data_params
        return PipelineSpec(method=method, resolved_configuration=configuration)

    def preprocessing_spec(self, schema: str) -> PreprocessingSpec:
        from .preprocessing import preprocessing_for_definition

        return preprocessing_for_definition(self, schema)

    def canonical_view(self, schema: str) -> dict[str, Any]:
        spec = self.preprocessing_spec(schema)
        payload = self.as_dict()
        payload["preprocessing"] = spec.as_dict()
        payload["detector"]["resolved_configuration"]["Data_Params"] = spec.to_legacy()[0]
        return payload

    @classmethod
    def from_canonical_view(cls, payload: Mapping[str, Any], schema: str) -> PipelineDefinition:
        from .preprocessing import PreprocessingSpec

        spec = PreprocessingSpec.from_dict(payload["preprocessing"], schema, allow_legacy=True)
        if spec.legacy is None or spec.legacy["execution_layer"] != "materialized_before_native":
            raise ValueError("PipelineDefinition requires its materialized legacy execution layout")
        data, operators = spec.to_legacy()
        detector = _thaw(payload["detector"])
        detector["resolved_configuration"]["Data_Params"] = data
        return cls.from_dict({**payload, "detector": detector, "preprocessing": [
            {"component_id": "native_preprocess", "parameters": {"value": data["preprocess"]}},
            {"component_id": "native_diff_order", "parameters": {"order": data["diff_order"]}},
            *operators,
        ]})


def pipeline_component_registry() -> dict[str, PipelineComponent]:
    components = {
        "native_preprocess": PipelineComponent(
            component_id="native_preprocess",
            stage="preprocessing",
            input_type="series",
            output_type="series",
            schemas=EASYTSAD_SCHEMAS,
            execution_adapter="easytsad_data_params",
            parameters={
                "value": {
                    "type": "string",
                    "legal_values": ["raw", "min-max", "z-score"],
                }
            },
        ),
        "native_diff_order": PipelineComponent(
            component_id="native_diff_order",
            stage="preprocessing",
            input_type="series",
            output_type="series",
            schemas=EASYTSAD_SCHEMAS,
            execution_adapter="easytsad_data_params",
            parameters={"order": {"type": "integer", "minimum": 0}},
        ),
        "robust_scaler": PipelineComponent(
            component_id="robust_scaler",
            stage="preprocessing",
            input_type="series",
            output_type="series",
            schemas=EASYTSAD_SCHEMAS,
            execution_adapter="materialized_dataset_view",
            parameters={},
        ),
        "moving_average": PipelineComponent(
            component_id="moving_average",
            stage="preprocessing",
            input_type="series",
            output_type="series",
            schemas=EASYTSAD_SCHEMAS,
            execution_adapter="materialized_dataset_view",
            parameters={"window": {"type": "integer", "minimum": 1, "default_value": 5}},
        ),
        "score_moving_average": PipelineComponent(
            component_id="score_moving_average",
            stage="postprocessing",
            input_type="score",
            output_type="score",
            schemas=EASYTSAD_SCHEMAS,
            execution_adapter="score_transform_before_evaluation",
            parameters={"window": {"type": "integer", "minimum": 1, "default_value": 5}},
        ),
        "mean_score_fusion": PipelineComponent(
            component_id="mean_score_fusion",
            stage="composition",
            input_type="score[]",
            output_type="score",
            schemas=EASYTSAD_SCHEMAS,
            execution_adapter="score_fusion",
            parameters={},
        ),
    }
    for method, spec in METHOD_CATALOG.items():
        if not spec.agent_available:
            continue
        schemas = tuple(
            schema
            for schema in EASYTSAD_SCHEMAS
            if spec.capability(schema).supported
        )
        if not schemas:
            continue
        components[f"detector:{method}"] = PipelineComponent(
            component_id=f"detector:{method}",
            stage="detector",
            input_type="series",
            output_type="score",
            schemas=schemas,
            execution_adapter="easytsad_method",
            parameters={
                "method": method,
                **({"supported_datasets": list(spec.supported_datasets)} if spec.supported_datasets is not None else {}),
                "artifact_type": spec.artifact_type,
                "fit_scope_by_schema": {
                    schema: spec.capability(schema).fit_scope for schema in schemas
                },
            },
        )
    return components


def pipeline_space_for_task(task: TaskContext) -> dict[str, Any]:
    stages: dict[str, list[dict[str, Any]]] = {
        "preprocessing": [],
        "representation": [],
        "detector": [],
        "postprocessing": [],
        "composition": [],
    }
    for component in pipeline_component_registry().values():
        if component.supports(task.schema):
            stages[component.stage].append(component.as_dict())
    for components in stages.values():
        components.sort(key=lambda item: item["component_id"])
    return {
        "schema": task.schema,
        "stages": stages,
        "prototype_limitations": [
            "non-native preprocessing requires a materialized dataset-view adapter",
            "postprocessing requires a score-transform-before-evaluation adapter",
            "composition requires a branch execution and score-fusion adapter",
        ],
    }


def validate_pipeline_definition(
    definition: PipelineDefinition,
    task: TaskContext,
) -> None:
    registry = pipeline_component_registry()
    for component in (
        *definition.preprocessing,
        *definition.representation,
        *definition.postprocessing,
    ):
        _require_registered_component(component, task.schema, registry)
    if definition.detector is not None:
        _require_registered_component(definition.detector, task.schema, registry)
    if definition.composition is not None:
        _require_registered_component(definition.composition, task.schema, registry)
    if definition.detector is None and definition.composition is None:
        raise ValueError("PipelineDefinition requires detector or composition")
    if definition.detector is not None and definition.composition is not None:
        raise ValueError("PipelineDefinition cannot have detector and composition together")


def _require_registered_component(
    component: Mapping[str, Any],
    schema: str,
    registry: Mapping[str, PipelineComponent],
) -> None:
    component_id = _text(component.get("component_id"), "component_id")
    registered = registry.get(component_id)
    if registered is None:
        raise ValueError(f"unknown Pipeline component: {component_id}")
    if not registered.supports(schema):
        raise ValueError(f"Pipeline component {component_id} is not executable for {schema}")


def _component_refs(values: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(_freeze_mapping(dict(value)) for value in values)


def _sequence(value: Any, label: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence")
    if not all(isinstance(item, Mapping) for item in value):
        raise TypeError(f"{label} must contain mappings")
    return tuple(value)


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("PipelineDefinition field must be a mapping or null")
    return _freeze_mapping(dict(value))


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not all(isinstance(key, str) for key in value):
        raise TypeError("PipelineDefinition keys must be strings")
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
    "PipelineComponent",
    "PipelineDefinition",
    "PipelineStage",
    "pipeline_component_registry",
    "pipeline_space_for_task",
    "validate_pipeline_definition",
]
