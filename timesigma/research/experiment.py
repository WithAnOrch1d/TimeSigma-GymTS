from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

from timesigma.algorithms.configuration import freeze_configuration
from timesigma.benchmark import runtime
from timesigma.benchmark.zero_shot import (
    load_zero_shot_membership,
    split_for_dataset,
)

from .pipeline import (
    NextAction,
    NormalizedPipeline,
    PipelineSpec,
    PlanningUnit,
    TaskContext,
    candidate_pipeline_from_action,
    resolve_baseline_pipeline,
)
from .pipeline_action import PipelineAction, apply_pipeline_action
from .pipeline_space import PipelineDefinition, pipeline_component_registry
from .pipeline_transforms import materialize_dataset_view


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_id: str
    task: TaskContext
    hypothesis: str
    baseline_reference: str
    baseline_pipeline: PipelineSpec | NormalizedPipeline
    action: NextAction
    seed: int
    budget: Mapping[str, Any]
    stopping_condition: str
    pipeline: PipelineSpec | NormalizedPipeline = field(init=False)

    def __post_init__(self) -> None:
        for label, value in (
            ("experiment_id", self.experiment_id),
            ("hypothesis", self.hypothesis),
            ("baseline_reference", self.baseline_reference),
            ("stopping_condition", self.stopping_condition),
        ):
            if not value.strip():
                raise ValueError(f"{label} is required")
        if not self.budget:
            raise ValueError("experiment budget is required")
        candidate = candidate_pipeline_from_action(
            self.baseline_pipeline,
            self.action,
            self.task,
        )
        object.__setattr__(self, "pipeline", candidate)
        object.__setattr__(
            self,
            "budget",
            MappingProxyType(deepcopy(dict(self.budget))),
        )

    @classmethod
    def from_baseline(
        cls,
        *,
        experiment_id: str,
        task: TaskContext,
        hypothesis: str,
        baseline_reference: str,
        baseline: Any,
        action: NextAction,
        seed: int,
        budget: Mapping[str, Any],
        stopping_condition: str,
    ) -> ExperimentPlan:
        _require_matching_case_identity(baseline, task, baseline_reference)
        return cls(
            experiment_id=experiment_id,
            task=task,
            hypothesis=hypothesis,
            baseline_reference=baseline_reference,
            baseline_pipeline=resolve_baseline_pipeline(baseline, task.dataset),
            action=action,
            seed=seed,
            budget=budget,
            stopping_condition=stopping_condition,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "task": self.task.as_dict(),
            "hypothesis": self.hypothesis,
            "baseline_reference": self.baseline_reference,
            "action": self.action.as_dict(),
            "pipeline": self.pipeline.as_dict(),
            "seed": self.seed,
            "budget": deepcopy(dict(self.budget)),
            "stopping_condition": self.stopping_condition,
        }


@dataclass(frozen=True)
class ExperimentRun:
    job_dir: Path
    results_dir: Path
    runtime_python: Path
    plan: ExperimentPlan

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_dir": str(self.job_dir),
            "results_dir": str(self.results_dir),
            "runtime_python": str(self.runtime_python),
            "experiment": self.plan.as_dict(),
        }


@dataclass(frozen=True)
class PipelineDefinitionExperimentPlan:
    experiment_id: str
    task: TaskContext
    hypothesis: str
    baseline_reference: str
    baseline_pipeline: PipelineDefinition
    action: PipelineAction
    seed: int
    budget: Mapping[str, Any]
    stopping_condition: str
    pipeline: PipelineDefinition = field(init=False)
    changed_research_dimension: str = field(init=False)
    detector_pipeline: PipelineSpec = field(init=False)

    def __post_init__(self) -> None:
        for label, value in (
            ("experiment_id", self.experiment_id),
            ("hypothesis", self.hypothesis),
            ("baseline_reference", self.baseline_reference),
            ("stopping_condition", self.stopping_condition),
        ):
            if not value.strip():
                raise ValueError(f"{label} is required")
        if not self.budget:
            raise ValueError("experiment budget is required")
        candidate = apply_pipeline_action(
            self.baseline_pipeline,
            self.action,
            self.task,
        )
        object.__setattr__(self, "pipeline", candidate)
        object.__setattr__(self, "changed_research_dimension", "preprocessing")
        object.__setattr__(
            self,
            "detector_pipeline",
            _detector_pipeline_for_execution(candidate),
        )
        object.__setattr__(
            self,
            "budget",
            MappingProxyType(deepcopy(dict(self.budget))),
        )

    @classmethod
    def from_baseline(
        cls,
        *,
        experiment_id: str,
        task: TaskContext,
        hypothesis: str,
        baseline_reference: str,
        baseline: Any,
        action: PipelineAction,
        seed: int,
        budget: Mapping[str, Any],
        stopping_condition: str,
    ) -> PipelineDefinitionExperimentPlan:
        _require_matching_case_identity(baseline, task, baseline_reference)
        return cls(
            experiment_id=experiment_id,
            task=task,
            hypothesis=hypothesis,
            baseline_reference=baseline_reference,
            baseline_pipeline=PipelineDefinition.from_pipeline_spec(
                resolve_baseline_pipeline(baseline, task.dataset),
            ),
            action=action,
            seed=seed,
            budget=budget,
            stopping_condition=stopping_condition,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "task": self.task.as_dict(),
            "hypothesis": self.hypothesis,
            "baseline_reference": self.baseline_reference,
            "pipeline_action": self.action.as_dict(),
            "baseline_pipeline": self.baseline_pipeline.as_dict(),
            "candidate_pipeline": self.pipeline.as_dict(),
            "changed_research_dimension": self.changed_research_dimension,
            "detector_pipeline": self.detector_pipeline.as_dict(),
            "seed": self.seed,
            "budget": deepcopy(dict(self.budget)),
            "stopping_condition": self.stopping_condition,
        }


@dataclass(frozen=True)
class PipelineDefinitionExperimentRun:
    job_dir: Path
    results_dir: Path
    runtime_python: Path
    plan: PipelineDefinitionExperimentPlan
    dataset_view_dir: Path | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_dir": str(self.job_dir),
            "results_dir": str(self.results_dir),
            "runtime_python": str(self.runtime_python),
            "dataset_view_dir": (
                None if self.dataset_view_dir is None else str(self.dataset_view_dir)
            ),
            "experiment": self.plan.as_dict(),
        }


@dataclass(frozen=True)
class PipelineRecommendation:
    recommendation_id: str
    task: TaskContext
    rationale: str
    action: Mapping[str, Any]
    seed: int
    budget: Mapping[str, Any]
    pipeline: PipelineSpec

    def __post_init__(self) -> None:
        for label, value in (
            ("recommendation_id", self.recommendation_id),
            ("rationale", self.rationale),
        ):
            if not value.strip():
                raise ValueError(f"{label} is required")
        if not self.budget:
            raise ValueError("recommendation budget is required")
        object.__setattr__(
            self,
            "action",
            MappingProxyType(deepcopy(dict(self.action))),
        )
        object.__setattr__(
            self,
            "budget",
            MappingProxyType(deepcopy(dict(self.budget))),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "task": self.task.as_dict(),
            "rationale": self.rationale,
            "action": deepcopy(dict(self.action)),
            "pipeline": self.pipeline.as_dict(),
            "seed": self.seed,
            "budget": deepcopy(dict(self.budget)),
        }


@dataclass(frozen=True)
class RecommendationRun:
    job_dir: Path
    results_dir: Path
    runtime_python: Path
    recommendation: PipelineRecommendation

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_dir": str(self.job_dir),
            "results_dir": str(self.results_dir),
            "runtime_python": str(self.runtime_python),
            "recommendation": self.recommendation.as_dict(),
        }


def run_pipeline_experiment(
    plan: ExperimentPlan,
    *,
    dataset_root: Path,
    output_dir: Path,
    dataset_source_root: Path | None = None,
    zero_shot_membership: Path | None = None,
) -> ExperimentRun:
    if plan.task.dataset.casefold() == "aiops":
        raise ValueError("AIOPS is forbidden in iterative research experiments")
    if plan.task.schema == "zero_shot" and zero_shot_membership is None:
        raise ValueError("zero_shot research requires frozen membership")
    if plan.task.schema == "zero_shot":
        assert zero_shot_membership is not None
        split = split_for_dataset(
            load_zero_shot_membership(zero_shot_membership),
            plan.task.dataset,
        )
        if (
            split.training_curves != plan.task.unit.training_curves
            or split.held_out_curves != plan.task.unit.held_out_curves
        ):
            raise ValueError(
                "zero-shot membership does not match the frozen planning unit"
            )

    job_dir = output_dir.resolve()
    if job_dir.exists() and any(job_dir.iterdir()):
        raise FileExistsError(f"experiment output already contains evidence: {job_dir}")
    job_dir.mkdir(parents=True, exist_ok=True)
    experiment_path = job_dir / "experiment.json"
    experiment_path.write_text(
        json.dumps(
            plan.as_dict(), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )

    normalized = plan.pipeline.normalized()
    if normalized.operators or normalized.ensemble_ref is not None:
        from .pipeline_execution import run_planned_pipeline

        results_dir, runtime_python = run_planned_pipeline(plan, dataset_root=dataset_root, output_dir=job_dir, dataset_source_root=dataset_source_root, zero_shot_membership=zero_shot_membership)
        return ExperimentRun(job_dir, results_dir, runtime_python, plan)

    with tempfile.TemporaryDirectory(prefix="timesigma-experiment-") as temporary:
        override_path = freeze_configuration(
            Path(temporary) / "override.toml",
            {
                plan.task.dataset: plan.pipeline.as_dict()[
                    "resolved_configuration"
                ]
            },
        )
        results_dir, runtime_python = runtime.run_method(
            method_name=plan.pipeline.method,
            schema=plan.task.schema,
            datasets=(plan.task.dataset,),
            dataset_root=dataset_root,
            dataset_source_root=dataset_source_root,
            output_dir=job_dir,
            configuration_name=plan.experiment_id,
            configuration_path=override_path,
            zero_shot_membership=zero_shot_membership,
            curve_names=(
                (plan.task.unit.curve,)
                if plan.task.schema == "naive"
                else None
            ),
            **({"ordered_members": {plan.task.dataset: plan.task.unit.training_curves}} if plan.task.schema == "all_in_one" else {}),
            seed=plan.seed,
            model_artifacts=True,
        )

    return ExperimentRun(
        job_dir=job_dir,
        results_dir=results_dir.resolve(),
        runtime_python=runtime_python.resolve(),
        plan=plan,
    )


def run_pipeline_definition_experiment(
    plan: PipelineDefinitionExperimentPlan,
    *,
    dataset_root: Path,
    dataset_dir: Path,
    output_dir: Path,
    dataset_source_root: Path | None = None,
    zero_shot_membership: Path | None = None,
    historical_replay: bool = False,
) -> PipelineDefinitionExperimentRun:
    """Replay the historical pre-split materialization layer explicitly."""
    if historical_replay is not True:
        raise ValueError("legacy materialized execution requires explicit historical_replay=True; use ExperimentPlan for current controlled research")
    from .planning import validate_planned_pipeline

    validate_planned_pipeline(plan.detector_pipeline.normalized(), plan.task)
    if plan.task.dataset.casefold() == "aiops":
        raise ValueError("AIOPS is forbidden in iterative research experiments")
    if plan.task.schema == "zero_shot" and zero_shot_membership is None:
        raise ValueError("zero_shot research requires frozen membership")
    if plan.task.schema == "zero_shot":
        assert zero_shot_membership is not None
        split = split_for_dataset(
            load_zero_shot_membership(zero_shot_membership),
            plan.task.dataset,
        )
        if (
            split.training_curves != plan.task.unit.training_curves
            or split.held_out_curves != plan.task.unit.held_out_curves
        ):
            raise ValueError(
                "zero-shot membership does not match the frozen planning unit"
            )

    job_dir = output_dir.resolve()
    if job_dir.exists() and any(job_dir.iterdir()):
        raise FileExistsError(f"pipeline experiment output already exists: {job_dir}")
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "pipeline-experiment.json").write_text(
        json.dumps(
            {**plan.as_dict(), "historical_replay": True, "execution_layer": "materialized_before_native"},
            ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )

    dataset_view_dir = None
    execution_dataset_root = dataset_root
    materialized_preprocessing = _materialized_preprocessing(plan.pipeline)
    if materialized_preprocessing:
        dataset_view_dir = materialize_dataset_view(
            dataset=plan.task.dataset,
            dataset_dir=dataset_dir,
            output_root=job_dir / "dataset-view",
            curve_names=_execution_curve_names(plan.task),
            preprocessing=materialized_preprocessing,
        )
        execution_dataset_root = dataset_view_dir

    with tempfile.TemporaryDirectory(prefix="timesigma-pipeline-experiment-") as temporary:
        override_path = freeze_configuration(
            Path(temporary) / "override.toml",
            {
                plan.task.dataset: plan.detector_pipeline.as_dict()[
                    "resolved_configuration"
                ]
            },
        )
        results_dir, runtime_python = runtime.run_method(
            method_name=plan.detector_pipeline.method,
            schema=plan.task.schema,
            datasets=(plan.task.dataset,),
            dataset_root=execution_dataset_root,
            dataset_source_root=dataset_source_root,
            output_dir=job_dir,
            configuration_name=plan.experiment_id,
            configuration_path=override_path,
            zero_shot_membership=zero_shot_membership,
            curve_names=(
                (plan.task.unit.curve,)
                if plan.task.schema == "naive"
                else None
            ),
            **({"ordered_members": {plan.task.dataset: plan.task.unit.training_curves}} if plan.task.schema == "all_in_one" else {}),
            seed=plan.seed,
            model_artifacts=True,
        )

    return PipelineDefinitionExperimentRun(
        job_dir=job_dir,
        results_dir=results_dir.resolve(),
        runtime_python=runtime_python.resolve(),
        plan=plan,
        dataset_view_dir=dataset_view_dir,
    )


def run_recommendation_evaluation(
    recommendation: PipelineRecommendation,
    *,
    dataset_root: Path,
    output_dir: Path,
    dataset_source_root: Path | None = None,
    zero_shot_membership: Path | None = None,
) -> RecommendationRun:
    if recommendation.task.dataset.casefold() == "aiops":
        raise ValueError("AIOPS is forbidden in one-shot recommendation evaluation")
    if recommendation.task.schema == "zero_shot" and zero_shot_membership is None:
        raise ValueError("zero_shot recommendation requires frozen membership")
    if recommendation.task.schema == "zero_shot":
        assert zero_shot_membership is not None
        split = split_for_dataset(
            load_zero_shot_membership(zero_shot_membership),
            recommendation.task.dataset,
        )
        if (
            split.training_curves != recommendation.task.unit.training_curves
            or split.held_out_curves != recommendation.task.unit.held_out_curves
        ):
            raise ValueError(
                "zero-shot membership does not match the frozen planning unit"
            )

    job_dir = output_dir.resolve()
    if not (job_dir / "recommendation.json").is_file():
        raise FileNotFoundError("recommendation must be frozen before evaluation")

    with tempfile.TemporaryDirectory(prefix="timesigma-recommendation-") as temporary:
        override_path = freeze_configuration(
            Path(temporary) / "override.toml",
            {
                recommendation.task.dataset: recommendation.pipeline.as_dict()[
                    "resolved_configuration"
                ]
            },
        )
        results_dir, runtime_python = runtime.run_method(
            method_name=recommendation.pipeline.method,
            schema=recommendation.task.schema,
            datasets=(recommendation.task.dataset,),
            dataset_root=dataset_root,
            dataset_source_root=dataset_source_root,
            output_dir=job_dir,
            configuration_name=recommendation.recommendation_id,
            configuration_path=override_path,
            zero_shot_membership=zero_shot_membership,
            curve_names=(
                (recommendation.task.unit.curve,)
                if recommendation.task.schema == "naive"
                else None
            ),
            **({"ordered_members": {recommendation.task.dataset: recommendation.task.unit.training_curves}} if recommendation.task.schema == "all_in_one" else {}),
            seed=recommendation.seed,            model_artifacts=True,
        )

    return RecommendationRun(
        job_dir=job_dir,
        results_dir=results_dir.resolve(),
        runtime_python=runtime_python.resolve(),
        recommendation=recommendation,
    )


def _detector_pipeline_for_execution(definition: PipelineDefinition) -> PipelineSpec:
    if definition.detector is None:
        raise ValueError("PipelineDefinition execution requires one detector")
    if definition.representation or definition.postprocessing or definition.composition is not None:
        raise ValueError("PipelineDefinition execution MVP only supports preprocessing")
    detector = dict(definition.detector)
    method = _text(detector.get("method"), "detector.method")
    configuration = _plain_json(_mapping(detector, "resolved_configuration"))
    data_params = _plain_json(_mapping(configuration, "Data_Params"))
    for component in definition.preprocessing:
        component_id = _text(component.get("component_id"), "component_id")
        parameters = _mapping(component, "parameters")
        if component_id == "native_preprocess":
            data_params["preprocess"] = _text(parameters.get("value"), "preprocess")
        elif component_id == "native_diff_order":
            order = parameters.get("order")
            if isinstance(order, bool) or not isinstance(order, int):
                raise TypeError("native diff_order must be an integer")
            data_params["diff_order"] = order
    configuration["Data_Params"] = data_params
    return PipelineSpec(method=method, resolved_configuration=configuration)


def _materialized_preprocessing(
    definition: PipelineDefinition,
) -> tuple[Mapping[str, Any], ...]:
    registry = pipeline_component_registry()
    selected = []
    for component in definition.preprocessing:
        component_id = _text(component.get("component_id"), "component_id")
        registered = registry.get(component_id)
        if registered is not None and registered.execution_adapter == "materialized_dataset_view":
            selected.append(_plain_json(component))
    return tuple(selected)


def _execution_curve_names(task: TaskContext) -> tuple[str, ...]:
    curves = (*task.unit.training_curves, *task.unit.held_out_curves)
    return tuple(dict.fromkeys(curves))


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    if isinstance(value, list):
        return [_plain_json(item) for item in value]
    return deepcopy(value)


def _require_matching_case_identity(
    baseline: Any,
    task: TaskContext,
    baseline_reference: str,
) -> None:
    if isinstance(baseline, PipelineSpec):
        return
    case_reference = (
        baseline.get("case_id")
        or baseline.get("experiment_id")
        or baseline.get("evidence_id")
        if isinstance(baseline, Mapping)
        else getattr(baseline, "case_id", None)
        or getattr(baseline, "experiment_id", None)
        or getattr(baseline, "evidence_id", None)
    )
    if case_reference is not None and case_reference != baseline_reference:
        raise ValueError(
            "baseline reference does not match the experiment: "
            f"{case_reference}"
        )
    for name, expected in (
        ("dataset", task.dataset),
        ("schema", task.schema),
    ):
        actual = (
            baseline.get(name)
            if isinstance(baseline, Mapping)
            else getattr(baseline, name, None)
        )
        if actual is not None and actual != expected:
            raise ValueError(
                f"baseline {name} does not match the experiment: {actual}"
            )
    _require_matching_planning_unit(baseline, task)


def _require_matching_planning_unit(baseline: Any, task: TaskContext) -> None:
    profile = (
        baseline.get("profile")
        if isinstance(baseline, Mapping)
        else getattr(baseline, "profile", None)
    )
    if isinstance(profile, Mapping):
        unit = PlanningUnit.from_dict(task.schema, profile)
        if unit != task.unit:
            raise ValueError("baseline planning unit does not match the experiment")
        return

    curve = (
        baseline.get("curve")
        if isinstance(baseline, Mapping)
        else getattr(baseline, "curve", None)
    )
    if curve is not None and curve != task.unit.curve:
        raise ValueError("baseline planning unit does not match the experiment")


__all__ = [
    "ExperimentPlan",
    "ExperimentRun",
    "PipelineDefinitionExperimentPlan",
    "PipelineDefinitionExperimentRun",
    "PipelineRecommendation",
    "RecommendationRun",
    "run_pipeline_definition_experiment",
    "run_pipeline_experiment",
    "run_recommendation_evaluation",
]
