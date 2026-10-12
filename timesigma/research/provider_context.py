from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
import re
from typing import Any, Literal

from timesigma.algorithms import METHOD_CATALOG
from timesigma.algorithms.configuration import resolve_configuration

from .canonical import ComponentEvidence, ResearchStepEvidence
from .collection import EvidenceCollection, plain_json
from .landscape import ComponentLandscape
from .pipeline import PREPROCESS_OPTIONS, NormalizedPipeline, PipelineSpec, TaskContext
from .pipeline_space import pipeline_component_registry
from .preprocessing import preprocessing_capability, preprocessing_search_space
from .state import ResearchState
from .semantics import context_window_parameter, execution_only_parameters, semantic_pipeline_diff, window_contract


ProviderRuntimeMode = Literal["offline_research", "one_shot"]


def decision_semantics(task: TaskContext) -> dict[str, Any]:
    """Interpret existing contracts without introducing another eligibility source."""
    return {
        "eligibility": "Use Catalog available_in_current_schema/schema_supported and Pipeline compatibility. Historical selection_status describes checkpoint selection, not whether a Method can be chosen or fitted on this task.",
        "artifact_selection": {
            "available": "The declared checkpoint selection policy succeeded.",
            "unavailable": "That policy could not run on the recorded unit; inspect its lifecycle and load contract.",
            "not_applicable": "No checkpoint selection policy exists in this lifecycle. Fitted estimators and replay recipes can be valid candidates.",
        },
        "new_fit_vs_load": "Choosing a Method/configuration for a new fit is not transferring another curve's fitted weights. Existing Artifact loading still requires its compatibility checks.",
        "schema": task.schema,
        "members": "Curve IDs are opaque, not compositions inferred from digits. NAIVE uses one curve's native train/validation/test segments; held_out_curves=[] is cross-curve membership, not evidence of in-sample test evaluation. Shared Schemas use their declared member lists.",
        "metrics": "Recorded official metrics remain observations of that exact configuration. Different metric definitions may disagree; F1=1 or low delay F1 alone does not invalidate a result. Missing records are unknown, not zero or inferior.",
        "training_signal": "FCVAE's real-label-free adapter retains native synthetic augmentation. This describes how its recorded results were generated, not a reason to discard them.",
        "transfer": "Profile similarity and source-task performance are evidence, not calibrated transfer probabilities or a guaranteed target ranking.",
    }


@dataclass(frozen=True)
class ProviderContextInputs:
    task: TaskContext
    profile: Mapping[str, Any]
    collection: EvidenceCollection
    component_landscape: ComponentLandscape
    research_state: ResearchState
    mode: ProviderRuntimeMode
    episode_feedback: Sequence[ResearchStepEvidence | Mapping[str, Any]] = ()
    relevant_history: Sequence[Mapping[str, Any]] = ()
    withheld_identity_values: Sequence[str] = ()
    selected_evidence_id: str | None = None
    memory_mode: str = "on"


def build_provider_context(inputs: ProviderContextInputs) -> dict[str, Any]:
    from .memory_access import MEMORY_TOOLS

    if inputs.mode not in {"offline_research", "one_shot"}:
        raise ValueError(f"unsupported Provider runtime mode: {inputs.mode}")
    if inputs.memory_mode not in {"on", "off"}:
        raise ValueError("memory_mode must be on or off")
    executable_capabilities = _executable_capabilities(inputs.task)
    context = {
        "task": _task_view(inputs.task, inputs.mode),
        "profile": _profile_view(inputs.profile),
        "component_landscape": _component_landscape_view(
            inputs.component_landscape,
            inputs.mode,
        ),
        "research_state": _research_state_view(
            inputs.collection,
            inputs.research_state,
            inputs.mode,
            recent_steps=tuple(item for item in inputs.episode_feedback[-2:] if isinstance(item, ResearchStepEvidence)),
            selected_evidence_id=inputs.selected_evidence_id,
        ),
        "episode_feedback": [
            _episode_feedback_view(item, inputs.task, inputs.collection) for item in inputs.episode_feedback[-2:]
        ],
        "relevant_history": [
            _relevant_history_view(item, inputs.mode) for item in inputs.relevant_history
        ],
        "executable_capabilities": [
            {key: item[key] for key in ("method", "schema_supported", "fit_scope", "allowed_action_dimensions")}
            for item in executable_capabilities
        ],
        "research_capabilities": _research_capabilities(
            inputs.task,
            inputs.mode,
            executable_capabilities,
        ),
        "memory_tools_available": list(MEMORY_TOOLS) if inputs.mode == "offline_research" else [],
        "decision_semantics": decision_semantics(inputs.task),
        "memory_visibility": {"mode": inputs.memory_mode if inputs.mode == "offline_research" else "off", "cross_task_access": inputs.mode == "offline_research" and inputs.memory_mode == "on", "note": "Memory OFF allows current-task facts only in research. Do not issue cross-task searches or reads when OFF; no-hit does not prove absence in the global historical store. Research Memory tools are not exposed by the one-shot view."},
    }
    _reject_absolute_paths(context)
    if inputs.mode == "one_shot":
        _reject_identity_values(
            context,
            (
                inputs.task.dataset,
                inputs.task.unit.curve,
                *inputs.task.unit.training_curves,
                *inputs.task.unit.held_out_curves,
                *inputs.withheld_identity_values,
            ),
        )
    return context


def _task_view(task: TaskContext, mode: ProviderRuntimeMode) -> dict[str, Any]:
    if mode == "offline_research":
        return task.as_dict()
    return {
        "schema": task.schema,
        "target_metric": task.target_metric,
        "input_state": task.input_state,
        "planning_scope": _planning_scope(task),
        "training_member_count": len(task.unit.training_curves),
        "held_out_member_count": len(task.unit.held_out_curves),
    }


def _planning_scope(task: TaskContext) -> str:
    if task.schema == "naive":
        return "curve"
    if task.schema == "zero_shot":
        return "zero_shot"
    return "dataset"


def _profile_view(profile: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_state": deepcopy(profile.get("input_state", "raw_train")),
        "curve_profile": deepcopy(profile.get("curve_profile")),
        "aggregate": deepcopy(profile.get("aggregate")),
    }
    split = profile.get("zero_shot_split")
    if isinstance(split, Mapping):
        training = split.get("training_curves")
        held_out = split.get("held_out_curves")
        enumerated = split.get("enumerated_curves")
        training_count = len(training) if isinstance(training, list) else 0
        held_out_count = len(held_out) if isinstance(held_out, list) else 0
        payload["zero_shot_split"] = {
            "proportion": split.get("proportion"),
            "enumerated_curve_count": (
                len(enumerated)
                if isinstance(enumerated, list)
                else training_count + held_out_count
            ),
            "training_curve_count": training_count,
            "held_out_curve_count": held_out_count,
        }
    else:
        payload["zero_shot_split"] = None
    return payload


def _component_landscape_view(
    landscape: ComponentLandscape,
    mode: ProviderRuntimeMode,
) -> dict[str, Any]:
    if mode == "one_shot":
        return {
            "visible": False,
            "reason": "current_task_component_metrics_withheld_before_freeze",
            "components": [],
        }
    return {
        "fact_type": "current_exact_task_component_evidence",
        "initial_baseline_policy": "Best known Component among already evaluated legal default Components for the current task under the frozen primary metric. This is a research starting point, not a Memory recommendation; it does not mean only one detector exists. The Runner-selected view may contain a subset of those records.",
        "components": [
            {
                "rank": index,
                "evidence_id": evidence.evidence_id,
                "method": str(evidence.pipeline["method"]),
                "pipeline": _pipeline_summary(evidence.pipeline, landscape.task.schema),
                "target_metric_value": _metric_value(
                    evidence.metrics,
                    landscape.task.target_metric,
                    evidence.evidence_id,
                ),
            }
            for index, evidence in enumerate(landscape.components, start=1)
        ],
        "best_value": landscape.best_value,
        "best_evidence_ids": list(landscape.best_evidence_ids),
        "default_initial_baseline_evidence_id": (
            landscape.default_baseline_evidence_id
        ),
    }


def _research_state_view(
    collection: EvidenceCollection,
    state: ResearchState,
    mode: ProviderRuntimeMode,
    *,
    recent_steps: Sequence[ResearchStepEvidence] = (),
    selected_evidence_id: str | None = None,
) -> dict[str, Any]:
    if mode == "one_shot":
        return {
            "visible": False,
            "reason": "iterative_research_state_not_exposed_before_one_shot_freeze",
        }
    steps = tuple(item for item in recent_steps if _matches_task(item.task, state.task))
    selected = collection.by_id(selected_evidence_id) if selected_evidence_id else None
    if selected is not None and not _matches_task(selected.task, state.task):
        raise ValueError("selected branch must belong to the current task")
    selected_pipeline = (
        selected.pipeline if isinstance(selected, ComponentEvidence)
        else selected.candidate_pipeline if isinstance(selected, ResearchStepEvidence)
        else None
    )
    return {
        "initial_baseline": (
            None
            if state.initial_baseline is None
            else {
                "evidence_id": state.initial_baseline["evidence_id"],
                "pipeline": _pipeline_summary(state.initial_baseline["pipeline"], state.task.schema),
                "target_value": state.initial_baseline["target_value"],
            }
        ),
        "global_best": {
            "value": state.global_best_value,
            "evidence_ids": list(state.global_best_evidence_ids),
            "incumbent_evidence_id": state.global_incumbent_evidence_id,
        },
        "research_step_count": state.research_step_count,
        "research_step_evidence_ids": [step.evidence_id for step in steps],
        "research_steps": [
            {
                "evidence_id": evidence.evidence_id,
                "baseline_evidence_id": evidence.baseline_evidence_id,
                "candidate_pipeline": _pipeline_summary(evidence.candidate_pipeline, state.task.schema),
                "action": _action_view(evidence.action),
                "target_metric_value": _metric_value(
                    evidence.metrics,
                    state.task.target_metric,
                    evidence.evidence_id,
                ),
            }
            for evidence in steps
        ],
        "lineage_edges": [dict(edge) for edge in state.lineage_edges if edge["child_evidence_id"] in {step.evidence_id for step in steps}],
        "history_truncated": state.research_step_count > len(steps),
        "history_access": "Use memory.search task_scope=current and memory.read to inspect other branches.",
        "selected_branch": None if selected is None else {
            "evidence_id": selected.evidence_id,
            "pipeline": _pipeline_summary(selected_pipeline, state.task.schema),
        },
        "runtime_budget": dict(state.runtime_budget),
        "stop_status": dict(state.stop_status),
    }


def _episode_feedback_view(
    item: ResearchStepEvidence | Mapping[str, Any],
    task: TaskContext,
    collection: EvidenceCollection,
) -> dict[str, Any]:
    if not isinstance(item, ResearchStepEvidence):
        return plain_json(item)
    if not _matches_task(item.task, task):
        raise ValueError("episode feedback must belong to the current task")
    return {
        "fact_type": "same_episode_research_step_feedback",
        "evidence_id": item.evidence_id,
        "baseline_evidence_id": item.baseline_evidence_id,
        "baseline_pipeline": _pipeline_summary(item.baseline_pipeline, task.schema),
        "candidate_pipeline": _pipeline_summary(item.candidate_pipeline, task.schema),
        "action": _action_view(item.action),
        "evaluation_facts": {
            "target_metric": task.target_metric,
            "before_value": item.before_value,
            "after_value": item.after_value,
            "edge_delta": item.edge_delta,
            "target_metric_value": _metric_value(
                item.metrics,
                task.target_metric,
                item.evidence_id,
            ),
            "baseline_official_metrics": dict(collection.by_id(item.baseline_evidence_id).metrics),
            "candidate_official_metrics": dict(item.metrics),
        },
        "semantic_diff": semantic_pipeline_diff(item.baseline_pipeline, item.candidate_pipeline, task),
        "recorded_model_reasoning": {
            "hypothesis": item.research_decision["hypothesis"],
            "diagnosis": plain_json(item.research_decision["diagnosis"]),
            "next_action": _action_view(item.research_decision["next_action"]),
            "next_action_rationale": item.research_decision[
                "next_action_rationale"
            ],
        },
    }


def _relevant_history_view(
    item: Mapping[str, Any],
    mode: ProviderRuntimeMode,
) -> dict[str, Any]:
    payload = plain_json(item)
    if mode == "offline_research":
        payload.setdefault("fact_type", "historical_other_task_evidence")
    return payload


def _executable_capabilities(task: TaskContext) -> list[dict[str, Any]]:
    capabilities: list[dict[str, Any]] = []
    for method in sorted(METHOD_CATALOG):
        spec = METHOD_CATALOG[method]
        capability = spec.capability(task.schema)
        if not capability.supported or not spec.agent_available or not spec.supports_dataset(task.dataset):
            continue
        default_pipeline = PipelineSpec.official_default(method, task.dataset)
        configuration = default_pipeline.as_dict()["resolved_configuration"]
        model_params = configuration["Model_Params"]
        window_parameter = context_window_parameter(method)
        hyperparameters = [
            _parameter_view(name, model_params[name])
            for name in sorted(model_params)
            if name != window_parameter
            and (spec.searchable_parameters is None or name in spec.searchable_parameters)
            and name not in execution_only_parameters()
            and not (method == "AnomalyTransformer" and name in {"step", "input_c", "output_c"})
        ]
        actions = ["algorithm", "preprocess", "diff_order"]
        if method == "IsolationForest":
            actions = ["algorithm"]
        window = None
        if window_parameter is not None and (spec.searchable_parameters is None or window_parameter in spec.searchable_parameters):
            window = _parameter_view(window_parameter, model_params[window_parameter])
            actions.append("window")
        if hyperparameters:
            actions.append("hyperparameter")
        capabilities.append(
            {
                "method": method,
                "schema_supported": True,
                "fit_scope": capability.fit_scope,
                "default_pipeline": _pipeline_summary(default_pipeline.as_dict(), task.schema),
                "allowed_action_dimensions": actions,
                "parameters": {
                    "preprocess": {
                        "legal_values": ["raw"] if method == "IsolationForest" else list(PREPROCESS_OPTIONS),
                        "default_value": configuration["Data_Params"]["preprocess"],
                    },
                    "diff_order": {
                        "minimum": 0,
                        **({"maximum": 0} if method == "IsolationForest" else {}),
                        "default_value": configuration["Data_Params"]["diff_order"],
                    },
                    "window": window,
                    "window_contract": window_contract(method),
                    "hyperparameters": hyperparameters,
                },
            }
        )
    return capabilities


def _research_capabilities(
    task: TaskContext,
    mode: ProviderRuntimeMode,
    executable_capabilities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if mode == "one_shot":
        return {
            "visible": False,
            "reason": "pipeline_research_actions_are_offline_only",
        }
    preprocessing = []
    for component in pipeline_component_registry().values():
        if (
            component.stage == "preprocessing"
            and component.execution_adapter == "materialized_dataset_view"
            and component.label_free
            and component.supports(task.schema)
        ):
            preprocessing.append(
                {
                    "operation": "add_component",
                    "stage": "preprocessing",
                    "component_id": component.component_id,
                    "parameters": deepcopy(dict(component.parameters)),
                }
            )
    preprocessing.sort(key=lambda item: item["component_id"])
    for item in preprocessing:
        if item["component_id"] == "moving_average":
            item["parameters"]["window"]["minimum"] = 2
    return {
        "detectors": [
            str(capability["method"])
            for capability in executable_capabilities
        ],
        "preprocessing": preprocessing,
        "pipeline_planning": {
            "action_dimension": "pipeline",
            "preprocessing": preprocessing_search_space(task.schema),
            "detector": "Catalog method and explicit parameter overrides; omitted parameters use Method defaults",
            "ensemble": {"methods": ["none", "mean"], "max_additional_detectors": 1, "supported_detectors": ["LOF", "OCSVM", "AR", "AE"], "normalization": "offline per-curve average rank", "alignment": "equal native suffix"},
        },
        "mvp_constraints": [
            "complete Pipeline proposals must differ from the baseline in exactly one research dimension",
            "legacy add_component keeps the detector unchanged",
            "one experiment must change exactly one research dimension",
            "mean fusion shares preprocessing and requires equal native score support",
        ],
    }


def _pipeline_summary(pipeline: Mapping[str, Any], schema: str | None = None) -> dict[str, Any]:
    normalized = NormalizedPipeline.from_dict(pipeline).canonical_payload()
    configuration = normalized["resolved_configuration"]
    summary = {
        "method": normalized["method"],
        "data_params": deepcopy(configuration["Data_Params"]),
        "model_params": deepcopy(configuration["Model_Params"]),
    }
    if schema is not None:
        summary["preprocessing"] = NormalizedPipeline.from_dict(pipeline).preprocessing_spec(schema).as_dict()
    if normalized["operators"] or normalized["ensemble_ref"] is not None:
        summary["operators"] = normalized["operators"]
        summary["ensemble_ref"] = normalized["ensemble_ref"]
    return summary


def lookup_capabilities(task: TaskContext, *, kind: str, name: str, one_shot: bool = False) -> dict[str, Any]:
    if task.dataset.casefold() == "aiops" and not one_shot:
        raise ValueError("AIOPS cannot enter research capability tools")
    if kind not in {"all", "detector", "preprocessing", "pipeline", "ensemble"}:
        raise ValueError("unknown capability kind")
    if not isinstance(name, str):
        raise TypeError("capability name must be a string")
    cards = []
    if kind in {"all", "detector"}:
        for method, spec in sorted(METHOD_CATALOG.items()):
            if not spec.agent_available or not spec.supports_dataset(task.dataset):
                continue
            cards.append({
                "kind": "detector", "name": method,
                "schema_support": {schema: capability.supported for schema, capability in spec.schemas.items()},
                "available_in_current_schema": spec.capability(task.schema).supported,
                **({"supported_datasets": list(spec.supported_datasets)} if spec.supported_datasets is not None else {}),
                **({"searchable_parameters": list(spec.searchable_parameters), "fixed_parameters": [name for name in PipelineSpec.official_default(method, task.dataset).resolved_configuration["Model_Params"] if name not in spec.searchable_parameters]} if spec.searchable_parameters is not None else {}),
                "preprocessing_compatibility": preprocessing_capability(method, task.schema),
                "default_pipeline": PipelineSpec.official_default(method, task.dataset).as_dict(),
                "window_parameter": context_window_parameter(method),
                "window_contract": window_contract(method),
                **({"label_free_execution": True, "artifact_selection": "not_applicable; train-only fitted estimator",
                    "score_definition": "-estimator.score_samples(X_test)",
                    "score_support_contract": {"start": 0, "stop": "N", "count": "N"},
                    "qualification_scope": "point-wise raw, context=1, seed=1, scikit-learn 1.9.0; all parameters and preprocessing frozen"}
                   if method == "IsolationForest" else {}),
                **({"label_free_execution": True,
                    "training_signal": "native synthetic anomaly/missing masks; no external anomaly labels",
                    "artifact_selection": "native validation monitor selected state; upstream default uses stopping state",
                    "stochastic_replay": "unchanged 10-step MCMC and 128 Monte Carlo draws; Artifact includes per-input torch RNG, repeated identical inputs require recorded occurrence order"} if method == "FCVAE" else {}),
                **({"score_support_contract": {"window_samples": "max(N-window_size-pred_len+1,0)", "score_length": "N-window_size-pred_len", "suffix_start": "window_size+pred_len", "native_tail_zeros": "pred_len-1", "minimum_test_length": "window_size+2*pred_len", "validation": "checked before model construction; research preview also checks file lengths"}, "label_free_execution": True, "artifact_selection": "best_unlabeled_validation; distinct from upstream final-state scoring"} if method in ("LSTMADalpha", "LSTMADbeta") else {}),
                "controlled_algorithm_change": {"preserves": ["Data_Params", "preprocessing components", "context window length", "sampling stride"], "detector_private_initialization": "target Method declared defaults; these initializations are not independent findings", "preview_tool": "experiment.preview"},
                "fixed_schema_parameters": {"input_c": 1, "output_c": 1, "step": 1} if method == "AnomalyTransformer" else {},
                "execution_only_parameters": sorted(execution_only_parameters() & set(PipelineSpec.official_default(method, task.dataset).resolved_configuration["Model_Params"])),
                "mean_fusion_alignment": None if method not in {"LOF", "OCSVM", "AR", "AE"} else {
                    "window_parameter": spec.sample_spec.window_parameter,
                    "horizon": spec.sample_spec.horizon,
                    "start_offset": "0" if spec.sample_spec.window_parameter is None else "window + horizon - 1",
                },
                "fit_scope": spec.capability(task.schema).fit_scope,
                "selection_objective": spec.capability(task.schema).selection_objective,
                "runtime": {"python_requires": spec.python_requires, "shared_runtime": spec.shared_runtime},
            })
    if kind in {"all", "preprocessing"}:
        cards.extend([
            {"kind": "preprocessing", "name": "none", "configuration": {"scaling": {"method": "raw", "fit_scope": "not_applicable"}, "temporal_transform": {"method": "none"}, "smoothing": {"method": "none"}}},
            {"kind": "preprocessing", "name": "differencing", "parameter": "temporal_transform", "legacy_parameter": "differencing", "type": "none or difference(order >= 1)", "execution": "native diff_order before scaling"},
            {"kind": "preprocessing", "name": "moving_average", "parameter": "smoothing", "legacy_parameter": "moving_average", "values": "none or moving_average(window >= 2)", "execution": "split-local centered edge padding; offline"},
            {"kind": "preprocessing", "name": "scaling", "parameter": "scaling", "values": preprocessing_search_space(task.schema)["scaling"], "execution": "Native min-max/z-score fit each curve train, including zero-shot held-out unlabeled train (target-specific unsupervised normalization). Robust fits only source training members in shared Schemas. Raw has no fit."},
        ])
        actions = _research_capabilities(task, "offline_research", ())["preprocessing"]
        for action in actions:
            matching = next((card for card in cards if card["name"] == action["component_id"]), None)
            if matching is None:
                cards.append({"kind": "preprocessing", "name": action["component_id"], "legacy_add_component": action})
            else:
                matching["legacy_add_component"] = action
    if kind in {"all", "pipeline"}:
        cards.append({
            "kind": "pipeline", "name": "pipeline",
            "preprocessing": preprocessing_search_space(task.schema),
            "actions": ["algorithm", "preprocess", "diff_order", "window", "hyperparameter", "pipeline"],
            "legacy_pipeline_action": "add_component for robust_scaler or moving_average",
            "order": ["native split", "differencing", "native scaling", "optional robust scaling", "optional moving average", "detector", "optional mean fusion"],
            "constraints": ["Exactly one actual research dimension changes", "No no-op or duplicate training", "No DAG", "Catalog defaults plus explicit overrides", "Other-task Evidence supplies hypotheses, not the executable baseline"],
            "algorithm_change": "Preserve shared preprocessing, context length and sampling stride; initialize target-private defaults. Incompatibilities are rejected, never reset. Preview before execution.",
            "semantic_axes": ["detector identity", "native scaling", "differencing", "context window", "sampling stride", "one private parameter", "one preprocessing component or its parameter", "preprocessing order", "ensemble branch addition/removal with existing branches frozen"],
        })
    if kind in {"all", "ensemble"}:
        cards.append({
            "kind": "ensemble", "name": "mean", "methods": ["none", "mean"],
            "maximum_detectors": 2, "compatible_detectors": [method for method in ("LOF", "OCSVM", "AR", "AE") if METHOD_CATALOG[method].capability(task.schema).supported],
            "normalization": "offline per-curve average rank before equal-weight mean",
            "constraints": ["Shared preprocessing", "Equal native suffix support: AR(p=k) aligns with AE(p=k+1); LOF and OCSVM both cover the full test series", "No identical duplicate detector configurations", "Each detector must support the current Schema", "Budget includes both detector executions", "Ensemble detector branches are not scientific parent nodes"],
        })
    if one_shot:
        for card in cards:
            if card["kind"] == "pipeline":
                card["constraints"] = ["One complete initial Pipeline; no target baseline transition", "No target execution or metrics during planning", "No DAG", "Pinned defaults plus explicit Agent overrides"]
    if name:
        cards = [card for card in cards if card["name"].casefold() == name.casefold()]
    return {"source": "Registry/Catalog capability, not experimental Evidence", "schema": task.schema, "items": cards,
            "artifact_note": decision_semantics(task)["eligibility"]}


def _parameter_view(name: str, value: Any) -> dict[str, Any]:
    return {
        "name": name,
        "primitive_type": _primitive_type(value),
        "default_value": deepcopy(value),
    }


def _primitive_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    return "object"


def _action_view(action: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "dimension": deepcopy(action.get("dimension")),
        "value": plain_json(action.get("value", action.get("after"))),
    }
    parameter = action.get("parameter")
    if parameter is not None:
        payload["parameter"] = deepcopy(parameter)
    return payload


def _metric_value(
    metrics: Mapping[str, Any],
    metric: str,
    evidence_id: str,
) -> float:
    value = metrics.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Evidence is missing target metric: {evidence_id}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Evidence target metric is not finite: {evidence_id}")
    return result


def _matches_task(evidence_task: Mapping[str, Any], task: TaskContext) -> bool:
    return (
        evidence_task["dataset"] == task.dataset
        and evidence_task["schema"] == task.schema
        and plain_json(evidence_task["unit"]) == plain_json(task.unit.as_dict())
    )


_ABSOLUTE_PATH_PATTERN = re.compile(r"(^/)|([A-Za-z]:[\\/])")


def _reject_absolute_paths(value: Any) -> None:
    if isinstance(value, str):
        if _ABSOLUTE_PATH_PATTERN.search(value):
            raise ValueError("Provider Context must not expose absolute paths")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_absolute_paths(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_absolute_paths(item)


def _reject_identity_values(
    value: Any,
    identity_values: Sequence[str | None],
) -> None:
    forbidden = tuple(
        sorted(
            (
                item
                for item in identity_values
                if isinstance(item, str) and item.strip()
            ),
            key=len,
            reverse=True,
        )
    )
    if not forbidden:
        return
    for text in _string_values(value):
        for item in forbidden:
            if (len(item) < 3 and text == item) or (len(item) >= 3 and item in text):
                raise ValueError("one-shot Provider Context exposes task identity")


def _string_values(value: Any) -> Sequence[str]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        strings: list[str] = []
        for item in value.values():
            strings.extend(_string_values(item))
        return tuple(strings)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        strings = []
        for item in value:
            strings.extend(_string_values(item))
        return tuple(strings)
    return ()


__all__ = [
    "ProviderContextInputs",
    "ProviderRuntimeMode",
    "build_provider_context",
    "lookup_capabilities",
]
