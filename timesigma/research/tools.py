from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

from timesigma.algorithms.artifacts import validate_score_lineage
from timesigma.algorithms import EASYTSAD_REVISION, METHOD_CATALOG
from timesigma.algorithms.configuration import resolve_configuration
from timesigma.benchmark.leaderboard import OFFICIAL_METRIC_IDS
from timesigma.benchmark.manifest import dataset_provenance
from timesigma.profiles import build_dataset_profiles

from .agent_memory import (
    _artifact_metadata as _artifact_metadata_for_agent_memory,
    validate_evidence_artifact,
)

from .canonical import (
    ComponentEvidence,
    ResearchStepEvidence,
    research_step_action_payload,
    research_step_evidence_id,
)
from .collection import (
    CanonicalEvidence,
    EvidenceCollection,
    canonical_pipeline_key_from_parts,
    evidence_directory,
    plain_json,
)
from .decision import (
    ResearchDecision,
    research_next_action_from_mapping,
)
from .experiment import (
    ExperimentPlan,
    PipelineDefinitionExperimentPlan,
    PipelineRecommendation,
    run_pipeline_definition_experiment,
    run_pipeline_experiment,
    run_recommendation_evaluation,
)
from .landscape import build_component_landscape
from .pipeline import (
    NextAction,
    NormalizedPipeline,
    PipelineSpec,
    PlanningUnit,
    TaskContext,
    candidate_pipeline_from_action,
    resolve_baseline_pipeline,
)
from .pipeline_action import PipelineAction
from .planning import infer_planning_action, normalize_operators, pipeline_detectors, plan_pipeline
from .provider_context import ProviderContextInputs, build_provider_context, lookup_capabilities
from .memory_access import read_memory, search_memory
from .evidence_facts import compare_evidence
from .retrieval import retrieve_component_history
from .result_parser import (
    _ParsedComponentJob,
    _ParsedComponentResult,
    parse_validated_component_results,
)
from .state import ResearchRuntimeSnapshot, build_research_state


_SCOPE_FIELDS = {
    "dataset",
    "schema",
    "target_metric",
    "dataset_root",
    "dataset_source_root",
    "dataset_dir",
    "dataset_source",
    "dataset_revision",
    "planning_unit",
    "profile_dir",
    "experiment_root",
    "zero_shot_membership",
    "experiment_index",
    "max_experiments",
    "max_wall_seconds",
}
_OPTIONAL_SCOPE_FIELDS = {
    "agent_memory_root",
    "memory_mode",
}
_RUN_EVIDENCE_FIELDS = {
    "experiment_id",
    "hypothesis",
    "baseline_evidence_id",
    "stopping_condition",
}
_RUN_ACTION_FIELDS = {"action", "pipeline_action"}
_PREPROCESSING_VALIDATION_FIELDS = {"enforce_preprocessing_categories"}
_FINISH_FIELDS = {
    "experiment_id",
    "diagnosis_status",
    "diagnosis",
    "next_action",
    "next_action_rationale",
}
_FINISH_EPISODE_FIELDS = {
    "reason",
    "rationale",
}
_RECOMMENDATION_FREEZE_FIELDS = {
    "recommendation_id",
    "method",
    "action",
    "rationale",
}
_RECOMMENDATION_EVALUATE_FIELDS = {
    "recommendation_id",
}
_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIAGNOSIS_STATUSES = {"supported", "refuted", "inconclusive"}
_FINISH_EPISODE_REASONS = {"target_saturated", "no_reasonable_next_step"}


@dataclass(frozen=True)
class _ResearchScope:
    task: TaskContext
    dataset_root: Path
    dataset_source_root: Path
    dataset_dir: Path
    dataset_source: str
    dataset_revision: str
    profile_dir: Path
    experiment_root: Path
    agent_memory_root: Path
    zero_shot_membership: Path | None
    experiment_index: int
    max_experiments: int
    max_wall_seconds: int
    memory_mode: str = "on"

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        extra_fields: set[str],
        tool: str,
        optional_extra_fields: set[str] | None = None,
    ) -> _ResearchScope:
        optional_fields = set(_OPTIONAL_SCOPE_FIELDS)
        if optional_extra_fields is not None:
            optional_fields.update(optional_extra_fields)
        _exact_fields(
            payload,
            _SCOPE_FIELDS | extra_fields,
            tool,
            optional=optional_fields,
        )
        dataset = _string(payload, "dataset")
        schema = _string(payload, "schema")
        unit = PlanningUnit.from_dict(schema, _mapping(payload, "planning_unit"))
        task = TaskContext(
            dataset=dataset,
            schema=schema,
            target_metric=_string(payload, "target_metric"),
            unit=unit,
        )
        if task.dataset.casefold() == "aiops":
            raise ValueError("AIOPS cannot enter iterative research")

        agent_memory_value = payload.get("agent_memory_root", "runs/agent-memory")
        if not isinstance(agent_memory_value, str):
            raise TypeError("agent_memory_root must be a string")
        memory_mode = payload.get("memory_mode", "on")
        if memory_mode not in {"on", "off"}:
            raise ValueError("memory_mode must be on or off")

        experiment_index = _positive_integer(payload, "experiment_index")
        max_experiments = _positive_integer(payload, "max_experiments")
        max_wall_seconds = _positive_integer(payload, "max_wall_seconds")
        if experiment_index > max_experiments:
            raise ValueError("experiment_index exceeds the frozen experiment budget")

        membership_value = payload.get("zero_shot_membership")
        if membership_value is not None and not isinstance(membership_value, str):
            raise TypeError("zero_shot_membership must be a string or null")
        membership = None if membership_value is None else Path(membership_value)
        if task.schema == "zero_shot" and membership is None:
            raise ValueError("zero_shot research requires frozen membership")
        if task.schema != "zero_shot" and membership is not None:
            raise ValueError("zero_shot membership requires schema zero_shot")

        dataset_root = Path(_string(payload, "dataset_root")).resolve()
        dataset_source_root = Path(
            _string(payload, "dataset_source_root")
        ).resolve()
        dataset_dir = Path(_string(payload, "dataset_dir")).resolve()
        if dataset_dir != (dataset_root / "UTS" / task.dataset).resolve():
            raise ValueError("dataset_dir does not match the execution dataset")
        try:
            dataset_dir.relative_to(dataset_source_root)
        except ValueError:
            raise ValueError(
                "dataset_dir is outside the frozen dataset source"
            ) from None
        source = _string(payload, "dataset_source")
        revision = _string(payload, "dataset_revision")
        actual_provenance = dataset_provenance(dataset_source_root)
        if (
            actual_provenance.get("source") != source
            or actual_provenance.get("revision") != revision
            or actual_provenance.get("clean") is not True
        ):
            raise ValueError(
                "dataset source provenance does not match the research scope"
            )
        return cls(
            task=task,
            dataset_root=dataset_root,
            dataset_source_root=dataset_source_root,
            dataset_dir=dataset_dir,
            dataset_source=source,
            dataset_revision=revision,
            profile_dir=Path(_string(payload, "profile_dir")).resolve(),
            experiment_root=Path(_string(payload, "experiment_root")),
            agent_memory_root=Path(agent_memory_value).resolve(),
            zero_shot_membership=membership,
            experiment_index=experiment_index,
            max_experiments=max_experiments,
            max_wall_seconds=max_wall_seconds,
            memory_mode=memory_mode,
        )


def dispatch_research_tool(
    tool: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    if tool == "memory.compare":
        scope = _ResearchScope.from_payload(payload, {"evidence_ids", "metric"}, tool,
                                            optional_extra_fields={"candidate_methods"})
        result = compare_evidence(
            _memory_collection(scope), scope.task, scope.agent_memory_root,
            evidence_ids=payload["evidence_ids"], metric=payload["metric"],
            candidate_methods=payload.get("candidate_methods"),
        )
        return {**result, "memory_visibility": _memory_visibility(scope)}
    if tool == "memory.search":
        scope = _ResearchScope.from_payload(payload, {"query", "filters", "order", "offset", "limit"}, tool)
        result = search_memory(
            _memory_collection(scope), scope.task,
            scope.agent_memory_root, _profile_for_scope(scope),
            query=payload["query"], filters=payload["filters"], order=payload["order"],
            offset=payload["offset"], limit=payload["limit"],
        )
        return {**result, "memory_visibility": _memory_visibility(scope)}
    if tool == "memory.read":
        scope = _ResearchScope.from_payload(payload, {"evidence_ids", "include_lineage", "offset", "limit"}, tool)
        result = read_memory(
            _memory_collection(scope), scope.task,
            scope.agent_memory_root, evidence_ids=payload["evidence_ids"],
            include_lineage=payload["include_lineage"], offset=payload["offset"], limit=payload["limit"],
        )
        return {**result, "memory_visibility": _memory_visibility(scope)}
    if tool == "capability.lookup":
        scope = _ResearchScope.from_payload(payload, {"kind", "name"}, tool)
        return lookup_capabilities(scope.task, kind=_string(payload, "kind"), name=payload["name"])
    if tool == "research.context":
        scope = _ResearchScope.from_payload(
            payload,
            set(),
            tool,
        )
        return _research_context(scope)
    if tool == "recommendation.context":
        scope = _ResearchScope.from_payload(
            payload,
            set(),
            tool,
        )
        return _recommendation_context(scope)
    if tool == "experiment.preview":
        scope = _ResearchScope.from_payload(payload, {"baseline_evidence_id"}, tool, optional_extra_fields=_RUN_ACTION_FIELDS | _PREPROCESSING_VALIDATION_FIELDS)
        return _preview_experiment(scope, payload)
    if tool == "experiment.run":
        scope = _ResearchScope.from_payload(
            payload,
            _RUN_EVIDENCE_FIELDS,
            tool,
            optional_extra_fields=_RUN_ACTION_FIELDS | {"preview_id"} | _PREPROCESSING_VALIDATION_FIELDS,
        )
        return _run_experiment(scope, payload)
    if tool == "research.complete_step":
        scope = _ResearchScope.from_payload(
            payload,
            _FINISH_FIELDS,
            tool,
        )
        return _complete_research_step(scope, payload)
    if tool == "research.finish_episode":
        scope = _ResearchScope.from_payload(
            payload,
            _FINISH_EPISODE_FIELDS,
            tool,
        )
        return _finish_episode(scope, payload)
    if tool == "recommendation.freeze":
        scope = _ResearchScope.from_payload(
            payload,
            _RECOMMENDATION_FREEZE_FIELDS,
            tool,
        )
        return _freeze_recommendation(scope, payload)
    if tool == "recommendation.evaluate":
        scope = _ResearchScope.from_payload(
            payload,
            _RECOMMENDATION_EVALUATE_FIELDS,
            tool,
        )
        return _evaluate_recommendation(scope, payload)
    raise ValueError(f"unsupported research tool: {tool}")


def _memory_collection(scope: _ResearchScope) -> EvidenceCollection:
    manifest = _read_mapping(scope.agent_memory_root / "manifest.json")
    source = manifest.get("dataset_sources", {}).get(
        scope.task.dataset, manifest.get("dataset_source"),
    )
    revision = manifest.get("dataset_revisions", {}).get(scope.task.dataset)
    if source != scope.dataset_source or revision != scope.dataset_revision:
        raise ValueError("Memory source does not match the frozen target data revision")
    collection = EvidenceCollection.from_directory(scope.agent_memory_root)
    if scope.memory_mode == "off":
        return EvidenceCollection(tuple(e for e in collection.evidence if _evidence_matches_task(e, scope)))
    return collection


def _memory_visibility(scope: _ResearchScope) -> dict[str, object]:
    return {"mode": scope.memory_mode, "cross_task_access": scope.memory_mode == "on", "note": "OFF permits only current-task facts in the Runner-selected view; do not repeat cross-task searches. ON permits independently visible history in that view."}


def _requested_action(scope: _ResearchScope, payload: Mapping[str, object], baseline: CanonicalEvidence) -> NextAction:
    if ("action" in payload) == ("pipeline_action" in payload):
        raise ValueError("requires exactly one of action or pipeline_action")
    if "action" in payload:
        return _action(payload, "action")
    component = _pipeline_action(payload, "pipeline_action")
    operators = _evidence_pipeline(baseline).normalized().canonical_payload()["operators"]
    operators.append({"component_id": component.component_id, "parameters": plain_json(component.parameters)})
    return NextAction("preprocessing", normalize_operators(operators))


def _category_validation(payload: Mapping[str, object]) -> bool:
    enabled = payload.get("enforce_preprocessing_categories", False)
    if not isinstance(enabled, bool):
        raise TypeError("enforce_preprocessing_categories must be boolean")
    return enabled


def _preview_binding(scope: _ResearchScope, baseline_id: str, action: NextAction, *, categories: bool = False) -> dict[str, object]:
    return {"task": scope.task.as_dict(), "dataset_source": scope.dataset_source, "dataset_revision": scope.dataset_revision, "memory_mode": scope.memory_mode, "baseline_evidence_id": baseline_id, "action": action.as_dict(), **({"preprocessing_contract": "categories"} if categories else {})}


def _preview_experiment(scope: _ResearchScope, payload: Mapping[str, object]) -> dict[str, object]:
    from .candidate_preview import preview_candidate, save_preview

    collection = _memory_collection(scope)
    baseline_id = _string(payload, "baseline_evidence_id")
    try:
        baseline = _active_evidence_baseline(collection, scope, baseline_id)
        action = _requested_action(scope, payload, baseline)
        result = preview_candidate(_evidence_pipeline(baseline), action, scope.task)
        if result["status"] == "ready":
            from timesigma.benchmark.input_contract import validate_research_input_lengths

            validate_research_input_lengths(NormalizedPipeline.from_dict(result["candidate_pipeline"]), scope.task, scope.dataset_dir)
        if result["status"] == "ready" and _category_validation(payload):
            from .preprocessing import validate_search_preprocessing

            validate_search_preprocessing(_evidence_pipeline(baseline), NormalizedPipeline.from_dict(result["candidate_pipeline"]), scope.task.schema)
        _require_initial_baseline_if_needed(collection, scope, baseline_id)
    except (ValueError, TypeError, KeyError) as error:
        return {"status": "incompatible", "reason": str(error), "baseline_evidence_id": baseline_id, "execution_budget": _execution_budget_view(scope)}
    result["baseline_evidence_id"] = baseline_id
    result["execution_budget"] = _execution_budget_view(scope)
    if result["status"] == "ready":
        result["preview_id"] = save_preview(scope.experiment_root, _preview_binding(scope, baseline_id, action, categories=_category_validation(payload)), result["candidate_pipeline"])
        reuse = collection.exact_pipeline_match(task=_task_payload(scope), pipeline=result["candidate_pipeline"])
        result["reuse_evidence_id"] = None if reuse is None else reuse.evidence_id
        if reuse is not None:
            result["detector_runs"] = 0
    return result


def _research_context(scope: _ResearchScope) -> dict[str, object]:
    collection = _memory_collection(scope)
    profile = _profile_for_scope(scope)
    landscape = build_component_landscape(collection, scope.task)
    research_state = build_research_state(
        collection,
        landscape,
        ResearchRuntimeSnapshot(
            experiments_used=_evidence_execution_attempt_count(scope),
            max_experiments=scope.max_experiments,
            max_wall_seconds=scope.max_wall_seconds,
        ),
    )
    attempts = _operation_attempts(scope)
    selected_id = landscape.default_baseline_evidence_id
    if attempts:
        selected_id = attempts[-1].get("committed_evidence_id") or attempts[-1]["baseline_evidence_id"]
    result = build_provider_context(
        ProviderContextInputs(
            task=scope.task,
            profile=profile,
            collection=collection,
            component_landscape=landscape,
            research_state=research_state,
            mode="offline_research",
            episode_feedback=_current_episode_feedback(scope, collection),
            relevant_history=(),
            selected_evidence_id=selected_id,
            memory_mode=scope.memory_mode,
        )
    )
    result["execution_budget"] = _execution_budget_view(scope)
    return result


def _recommendation_context(scope: _ResearchScope) -> dict[str, object]:
    profile = _profile_for_scope(scope)
    return {
        "task_contract": _recommendation_task_contract(scope),
        "profile": _recommendation_profile_view(scope, profile),
        "catalog": _recommendation_catalog_view(scope),
        "historical_pipeline_evidence": [],
    }


def _profile_for_scope(scope: _ResearchScope) -> dict[str, object]:
    profiles = build_dataset_profiles(
        dataset=scope.task.dataset,
        dataset_dir=scope.dataset_dir,
        schema=scope.task.schema,
        provenance={
            "source": scope.dataset_source,
            "revision": scope.dataset_revision,
        },
        zero_shot_membership=scope.zero_shot_membership,
        profile_dir=scope.profile_dir,
    )
    views = [profile.as_dict() for profile in profiles]
    if scope.task.schema == "all_in_one":
        for view in views:
            if set(view["training_curves"]) == set(scope.task.unit.training_curves):
                # The curve Profile/aggregate is unchanged; task authority owns execution order.
                view["training_curves"] = list(scope.task.unit.training_curves)
    matches = tuple(view for view in views if _profile_matches_unit(view, scope))
    if len(matches) != 1:
        raise ValueError("recommendation context requires one frozen Profile view")
    return matches[0]


def _profile_matches_unit(
    profile: Mapping[str, object],
    scope: _ResearchScope,
) -> bool:
    unit = scope.task.unit
    return (
        profile.get("curve") == unit.curve
        and tuple(profile.get("training_curves", ())) == unit.training_curves
        and tuple(profile.get("held_out_curves", ())) == unit.held_out_curves
    )


def _recommendation_profile_view(
    scope: _ResearchScope,
    profile: Mapping[str, object],
) -> dict[str, object]:
    return _recommendation_profile_summary(profile)


def _recommendation_task_contract(scope: _ResearchScope) -> dict[str, object]:
    return {
        "schema": scope.task.schema,
        "target_metric": scope.task.target_metric,
        "input_state": scope.task.input_state,
        "planning_scope": _planning_scope(scope.task),
        "training_member_count": len(scope.task.unit.training_curves),
        "held_out_member_count": len(scope.task.unit.held_out_curves),
        "one_shot_constraints": {
            "current_task_metrics": "withheld",
            "current_task_pipeline_evidence": "withheld",
            "exact_self_evidence": "forbidden",
            "current_task_capability_results": "not_required",
            "evaluation_feedback": (
                "Freeze exactly one recommendation before evaluation; the "
                "evaluation result is outside the Agent session."
            ),
            "pipeline_contract": (
                "Choose one Catalog-supported method at its official default, "
                "or apply exactly one preprocess/diff_order/window/hyperparameter "
                "override to that method's official default."
            ),
        },
    }


def _planning_scope(task: TaskContext) -> str:
    if task.schema == "naive":
        return "curve"
    if task.schema == "zero_shot":
        return "zero_shot"
    return "dataset"


def _recommendation_profile_summary(
    profile: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "input_state": deepcopy(profile.get("input_state", "raw_train")),
        "curve_profile": deepcopy(profile.get("curve_profile")),
        "aggregate": deepcopy(profile.get("aggregate")),
    }
    split = profile.get("zero_shot_split")
    if isinstance(split, Mapping):
        training = split.get("training_curves")
        held_out = split.get("held_out_curves")
        training_count = len(training) if isinstance(training, list) else 0
        held_out_count = len(held_out) if isinstance(held_out, list) else 0
        enumerated = split.get("enumerated_curves")
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


_CONSUMED_ATTEMPT_STATUSES = {
    "allocated",
    "running",
    "succeeded_pending_diagnosis",
    "failed",
    "committed",
}


def _run_evidence_first_experiment(
    scope: _ResearchScope,
    payload: Mapping[str, object],
) -> dict[str, object]:
    experiment_id = _experiment_id(payload)
    hypothesis = _string(payload, "hypothesis")
    baseline_evidence_id = _string(payload, "baseline_evidence_id")
    action = _action(payload, "action")
    stopping_condition = _string(payload, "stopping_condition")

    request = {key: plain_json(payload[key]) for key in ("hypothesis", "baseline_evidence_id", "action", "stopping_condition")}
    if _category_validation(payload):
        request["preprocessing_contract"] = "categories"
    request["scope"] = {"task": scope.task.as_dict(), "dataset_source": scope.dataset_source, "dataset_revision": scope.dataset_revision, "max_experiments": scope.max_experiments, "max_wall_seconds": scope.max_wall_seconds}
    attempt = _operation_attempt(scope, experiment_id)
    if attempt is not None and attempt.get("request") is not None and attempt["request"] != request:
        raise ValueError("experiment_id conflicts with its frozen request")

    collection = _memory_collection(scope)
    if attempt is not None and attempt["status"] == "committed" and "preview_id" not in payload:
        return _recover_evidence_first_attempt(scope, attempt)
    _require_initial_baseline_if_needed(collection, scope, baseline_evidence_id)
    baseline = _active_evidence_baseline(collection, scope, baseline_evidence_id)
    baseline_pipeline = _evidence_pipeline(baseline)
    candidate = candidate_pipeline_from_action(baseline_pipeline, action, scope.task)
    from timesigma.benchmark.input_contract import validate_research_input_lengths

    validate_research_input_lengths(candidate, scope.task, scope.dataset_dir)
    if _category_validation(payload):
        from .preprocessing import validate_search_preprocessing

        validate_search_preprocessing(baseline_pipeline, candidate, scope.task.schema)
    if "preview_id" in payload:
        from .candidate_preview import check_preview

        if not check_preview(scope.experiment_root, payload["preview_id"], _preview_binding(scope, baseline_evidence_id, action, categories=_category_validation(payload)), candidate.normalized().canonical_payload()):
            return {"status": "rejected", "reason": "Preview does not match this scope, baseline and action; request a new experiment.preview before execution", "execution_budget": _execution_budget_view(scope)}
        if attempt is not None and NormalizedPipeline.from_dict(attempt["candidate_pipeline"]) != candidate.normalized():
            return {"status": "rejected", "reason": "This experiment_id belongs to a different archived resolved Pipeline; inspect its Evidence or choose a new experiment_id", "execution_budget": _execution_budget_view(scope)}
    if action.dimension == "pipeline":
        action = infer_planning_action(baseline_pipeline.normalized(), candidate.normalized(), scope.task)
    candidate_pipeline = candidate.normalized().canonical_payload()
    if attempt is not None and NormalizedPipeline.from_dict(attempt["candidate_pipeline"]).canonical_payload() != candidate_pipeline:
        return {"status": "rejected", "reason": "Archived attempt resolves to a different Pipeline under the current contract; no automatic re-execution or new admission is allowed. Inspect its recorded facts or use a new experiment_id.", "execution_budget": _execution_budget_view(scope)}
    existing = collection.exact_pipeline_match(
        task=_task_payload(scope),
        pipeline=candidate_pipeline,
    )
    if existing is not None:
        return _reuse_evidence_result(
            scope=scope,
            baseline=baseline,
            reused=existing,
            action=action,
            hypothesis=hypothesis,
        )

    if attempt is not None:
        return _recover_evidence_first_attempt(scope, attempt)
    duplicate_attempt = _operation_attempt_with_pipeline(
        scope,
        candidate_pipeline,
        exclude_experiment_id=experiment_id,
    )
    if duplicate_attempt is not None:
        return _recover_or_reject_duplicate_pipeline_attempt(
            scope,
            duplicate_attempt,
        )

    if _evidence_execution_attempt_count(scope) >= scope.max_experiments:
        raise ValueError("experiment execution budget is exhausted")
    if any(item.get("status") == "succeeded_pending_diagnosis" for item in _operation_attempts(scope)):
        raise ValueError("complete the pending diagnosis before another experiment")

    experiment_dir = scope.experiment_root / experiment_id
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise ValueError(
            "run-local experiment output already exists without an operation record"
        )
    plan = ExperimentPlan.from_baseline(
        experiment_id=experiment_id,
        task=scope.task,
        hypothesis=hypothesis,
        baseline_reference=baseline_evidence_id,
        baseline=baseline_pipeline,
        action=action,
        seed=1,
        budget={"max_runs": len(pipeline_detectors(candidate.normalized())), "max_wall_seconds": scope.max_wall_seconds},
        stopping_condition=stopping_condition,
    )
    _upsert_operation_attempt(
        scope,
        {
            "experiment_id": experiment_id,
            "status": "allocated",
            "request": request,
            "branch_runs": len(pipeline_detectors(candidate.normalized())),
            "baseline_evidence_id": baseline_evidence_id,
            "hypothesis": hypothesis,
            "action": action.as_dict(),
            "baseline_pipeline": baseline_pipeline.as_dict(),
            "candidate_pipeline": candidate_pipeline,
            "stopping_condition": stopping_condition,
            "output_dir": str(experiment_dir),
        },
    )
    _upsert_operation_attempt(
        scope,
        {**_operation_attempt(scope, experiment_id), "status": "running"},  # type: ignore[arg-type]
    )
    try:
        outcome = run_pipeline_experiment(
            plan,
            dataset_root=scope.dataset_root,
            dataset_source_root=scope.dataset_source_root,
            output_dir=experiment_dir,
            zero_shot_membership=scope.zero_shot_membership,
        )
        parsed_result = _evidence_first_parsed_result(
            scope=scope,
            plan=plan,
            experiment_dir=outcome.job_dir,
        )
    except Exception:
        attempt = _operation_attempt(scope, experiment_id)
        if attempt is not None:
            _upsert_operation_attempt(scope, {**attempt, "status": "failed"})
        raise

    result = _new_execution_result(
        scope=scope,
        experiment_id=experiment_id,
        baseline=baseline,
        baseline_pipeline=baseline_pipeline.as_dict(),
        candidate_pipeline=candidate_pipeline,
        action=action,
        hypothesis=hypothesis,
        parsed_result=parsed_result,
        status="executed_pending_diagnosis",
    )
    attempt = _operation_attempt(scope, experiment_id)
    assert attempt is not None
    _upsert_operation_attempt(
        scope,
        {
            **attempt,
            "status": "succeeded_pending_diagnosis",
            "official_metrics": result["official_metrics"],
            "target_metric": result["target_metric"],
            "baseline_value": result["baseline_value"],
            "candidate_value": result["candidate_value"],
            "delta": result["delta"],
        },
    )
    return result


def _recover_evidence_first_attempt(
    scope: _ResearchScope,
    attempt: Mapping[str, object],
) -> dict[str, object]:
    status = _string(attempt, "status")
    collection = EvidenceCollection.from_directory(scope.agent_memory_root)
    committed_id = attempt.get("committed_evidence_id")
    if status == "committed" and isinstance(committed_id, str):
        return _committed_evidence_result(scope, collection.by_id(committed_id))
    if status in {"allocated", "running"}:
        return _resume_evidence_first_attempt(scope, attempt, collection)
    if status != "succeeded_pending_diagnosis":
        raise ValueError(f"cannot recover incomplete operation status: {status}")
    return _execution_result_from_attempt(scope, attempt, collection)


def _resume_evidence_first_attempt(
    scope: _ResearchScope,
    attempt: Mapping[str, object],
    collection: EvidenceCollection,
) -> dict[str, object]:
    status = _string(attempt, "status")
    experiment_dir = Path(_string(attempt, "output_dir"))
    plan = _attempt_plan(scope, attempt)
    recovered_result = _try_recover_parsed_result(scope, plan, experiment_dir)
    if recovered_result is not None:
        result = _execution_result_from_attempt(
            scope,
            {
                **attempt,
                "status": "succeeded_pending_diagnosis",
                "official_metrics": _parsed_metrics(recovered_result),
            },
            collection,
            parsed_result=recovered_result,
        )
        _upsert_operation_attempt(
            scope,
            {
                **attempt,
                "status": "succeeded_pending_diagnosis",
                "official_metrics": result["official_metrics"],
                "target_metric": result["target_metric"],
                "baseline_value": result["baseline_value"],
                "candidate_value": result["candidate_value"],
                "delta": result["delta"],
            },
        )
        return result

    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        _upsert_operation_attempt(scope, {**attempt, "status": "failed"})
        raise ValueError(
            f"cannot recover incomplete operation status: {status}"
        )

    _upsert_operation_attempt(scope, {**attempt, "status": "running"})
    try:
        outcome = run_pipeline_experiment(
            plan,
            dataset_root=scope.dataset_root,
            dataset_source_root=scope.dataset_source_root,
            output_dir=experiment_dir,
            zero_shot_membership=scope.zero_shot_membership,
        )
        parsed_result = _evidence_first_parsed_result(
            scope=scope,
            plan=plan,
            experiment_dir=outcome.job_dir,
        )
    except Exception:
        current = _operation_attempt(scope, _string(attempt, "experiment_id"))
        if current is not None:
            _upsert_operation_attempt(scope, {**current, "status": "failed"})
        raise

    result = _execution_result_from_attempt(
        scope,
        {**attempt, "status": "succeeded_pending_diagnosis"},
        collection,
        parsed_result=parsed_result,
    )
    _upsert_operation_attempt(
        scope,
        {
            **attempt,
            "status": "succeeded_pending_diagnosis",
            "official_metrics": result["official_metrics"],
            "target_metric": result["target_metric"],
            "baseline_value": result["baseline_value"],
            "candidate_value": result["candidate_value"],
            "delta": result["delta"],
        },
    )
    return result


def _try_recover_parsed_result(
    scope: _ResearchScope,
    plan: ExperimentPlan,
    experiment_dir: Path,
) -> _ParsedComponentResult | None:
    if not experiment_dir.exists():
        return None
    try:
        return _evidence_first_parsed_result(
            scope=scope,
            plan=plan,
            experiment_dir=experiment_dir,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None


def _execution_result_from_attempt(
    scope: _ResearchScope,
    attempt: Mapping[str, object],
    collection: EvidenceCollection,
    *,
    parsed_result: _ParsedComponentResult | None = None,
) -> dict[str, object]:
    experiment_id = _string(attempt, "experiment_id")
    baseline = _active_evidence_baseline(
        collection,
        scope,
        _string(attempt, "baseline_evidence_id"),
    )
    action = _attempt_action(attempt)
    plan = _attempt_plan(scope, attempt)
    if parsed_result is None:
        parsed_result = _evidence_first_parsed_result(
            scope=scope,
            plan=plan,
            experiment_dir=Path(_string(attempt, "output_dir")),
        )
    return _new_execution_result(
        scope=scope,
        experiment_id=experiment_id,
        baseline=baseline,
        baseline_pipeline=_mapping(attempt, "baseline_pipeline"),
        candidate_pipeline=_mapping(attempt, "candidate_pipeline"),
        action=action,
        hypothesis=_string(attempt, "hypothesis"),
        parsed_result=parsed_result,
        status="executed_pending_diagnosis",
    )


def _attempt_plan(scope: _ResearchScope, attempt: Mapping[str, object]) -> ExperimentPlan:
    return ExperimentPlan.from_baseline(
        experiment_id=_string(attempt, "experiment_id"),
        task=scope.task,
        hypothesis=_string(attempt, "hypothesis"),
        baseline_reference=_string(attempt, "baseline_evidence_id"),
        baseline=_mapping(attempt, "baseline_pipeline"),
        action=_attempt_action(attempt),
        seed=1,
        budget={"max_runs": attempt.get("branch_runs", 1), "max_wall_seconds": scope.max_wall_seconds},
        stopping_condition=_string(attempt, "stopping_condition"),
    )


def _attempt_action(attempt: Mapping[str, object]) -> NextAction:
    raw_action = _mapping(attempt, "action")
    return NextAction(
        dimension=_string(raw_action, "dimension"),  # type: ignore[arg-type]
        parameter=_optional_action_parameter(raw_action),
        value=deepcopy(raw_action.get("value")),
    )


def _complete_evidence_first_research_step(
    scope: _ResearchScope,
    payload: Mapping[str, object],
) -> dict[str, object]:
    experiment_id = _experiment_id(payload)
    attempt = _operation_attempt(scope, experiment_id)
    if attempt is None:
        raise ValueError("ResearchStep completion requires a pending execution attempt")
    collection = EvidenceCollection.from_directory(scope.agent_memory_root)
    status = _string(attempt, "status")
    if status == "committed":
        committed = _string(attempt, "committed_evidence_id")
        evidence = collection.by_id(committed)
        _require_research_decision_matches(evidence, payload)
        return _committed_evidence_result(scope, evidence)
    if status != "succeeded_pending_diagnosis":
        raise ValueError(f"ResearchStep completion cannot use status: {status}")

    diagnosis_status = _string(payload, "diagnosis_status")
    if diagnosis_status not in _DIAGNOSIS_STATUSES:
        raise ValueError(f"unsupported diagnosis_status: {diagnosis_status}")
    diagnosis = _string(payload, "diagnosis")
    next_action = research_next_action_from_mapping(payload, "next_action")
    next_action_rationale = _string(payload, "next_action_rationale")
    decision = ResearchDecision(
        hypothesis=_string(attempt, "hypothesis"),
        diagnosis_verdict=diagnosis_status,
        diagnosis_reason=diagnosis,
        next_action=next_action,
        next_action_rationale=next_action_rationale,
    )

    baseline = _active_evidence_baseline(
        collection,
        scope,
        _string(attempt, "baseline_evidence_id"),
    )
    action = _attempt_action(attempt)
    from .pipeline import validate_declared_pipeline_change

    validate_declared_pipeline_change(_mapping(attempt, "baseline_pipeline"), action, _mapping(attempt, "candidate_pipeline"), scope.task)
    plan = _attempt_plan(scope, attempt)
    parsed_result = _evidence_first_parsed_result(
        scope=scope,
        plan=plan,
        experiment_dir=Path(_string(attempt, "output_dir")),
    )
    metrics = _parsed_metrics(parsed_result)
    baseline_value = _target_value_for_evidence(baseline, scope.task.target_metric)
    candidate_value = metrics[scope.task.target_metric]
    edge_action = research_step_action_payload(
        baseline_pipeline=_mapping(attempt, "baseline_pipeline"),
        candidate_pipeline=_mapping(attempt, "candidate_pipeline"),
        action=action,
    )
    artifact_summary, artifact_metadata, artifact_source = _research_artifact_from_reference(
        parsed_result.artifact_reference,
        Path(_string(attempt, "output_dir")),
    )
    evidence = ResearchStepEvidence(
        evidence_id=research_step_evidence_id(
            task=_task_payload(scope),
            experiment_id=experiment_id,
        ),
        task=_task_payload(scope),
        target_metric=scope.task.target_metric,
        baseline_evidence_id=_string(attempt, "baseline_evidence_id"),
        baseline_pipeline=_mapping(attempt, "baseline_pipeline"),
        candidate_pipeline=_mapping(attempt, "candidate_pipeline"),
        action=edge_action,
        before_value=baseline_value,
        after_value=candidate_value,
        edge_delta=candidate_value - baseline_value,
        metrics=metrics,
        research_decision=decision.as_dict(),
        artifact=artifact_summary,
    )
    committed_id = _commit_research_step_evidence(
        scope=scope,
        evidence=evidence,
        artifact_metadata=artifact_metadata,
        artifact_source=artifact_source,
        **({"pipeline_bundle": Path(_string(attempt, "output_dir"))} if parsed_result.artifact_reference.get("pipeline_bundle") else {}),
    )
    _upsert_operation_attempt(
        scope,
        {
            **attempt,
            "status": "committed",
            "committed_evidence_id": committed_id,
        },
    )
    return _committed_evidence_result(
        scope,
        EvidenceCollection.from_directory(scope.agent_memory_root).by_id(committed_id),
    )


def _evidence_first_parsed_result(
    *,
    scope: _ResearchScope,
    plan: ExperimentPlan,
    experiment_dir: Path,
) -> _ParsedComponentResult:
    normalized = plan.pipeline.normalized()
    if normalized.operators or normalized.ensemble_ref is not None:
        from .pipeline_execution import parse_planned_result

        return parse_planned_result(plan, experiment_dir, scope.dataset_dir, {"source": scope.dataset_source, "revision": scope.dataset_revision}, scope.profile_dir)
    parsed = parse_validated_component_results(
        experiment_dir,
        scope.dataset_dir,
        {
            "source": scope.dataset_source,
            "revision": scope.dataset_revision,
        },
        profile_dir=scope.profile_dir,
    )
    result = _parsed_result_for_unit(parsed, scope)
    _require_parsed_scope(parsed, result, scope)
    _require_parsed_candidate_identity(parsed, plan)
    return result


def _pipeline_definition_parsed_result(
    *,
    scope: _ResearchScope,
    plan: PipelineDefinitionExperimentPlan,
    experiment_dir: Path,
) -> _ParsedComponentResult:
    parsed = parse_validated_component_results(
        experiment_dir,
        scope.dataset_dir,
        {
            "source": scope.dataset_source,
            "revision": scope.dataset_revision,
        },
        profile_dir=scope.profile_dir,
    )
    result = _parsed_result_for_unit(parsed, scope)
    _require_parsed_scope(parsed, result, scope)
    _require_parsed_pipeline_definition_identity(parsed, plan)
    return result


def _parsed_result_for_unit(
    parsed: _ParsedComponentJob,
    scope: _ResearchScope,
) -> _ParsedComponentResult:
    selected = tuple(unit for unit in parsed.units if _parsed_result_matches_unit(unit, scope))
    if len(selected) != 1 or len(parsed.units) != 1:
        raise ValueError(
            "Pipeline experiment must produce exactly its frozen Planning Unit"
        )
    return selected[0]


def _parsed_result_matches_unit(
    result: _ParsedComponentResult,
    scope: _ResearchScope,
) -> bool:
    unit = scope.task.unit
    return (
        result.curve == unit.curve
        and tuple(result.training_curves) == unit.training_curves
        and tuple(result.held_out_curves) == unit.held_out_curves
    )


def _require_parsed_scope(
    parsed: _ParsedComponentJob,
    result: _ParsedComponentResult,
    scope: _ResearchScope,
) -> None:
    if (
        parsed.dataset != scope.task.dataset
        or parsed.schema != scope.task.schema
        or not _parsed_result_matches_unit(result, scope)
    ):
        raise ValueError("Component result is outside the fixed research scope")
    evidence = parsed.evidence_reference
    if (
        evidence.get("dataset_source") != scope.dataset_source
        or evidence.get("dataset_revision") != scope.dataset_revision
        or evidence.get("easytsad_revision") != EASYTSAD_REVISION
        or evidence.get("seed") != 1
    ):
        raise ValueError(
            "Component result lineage is outside the fixed research scope"
        )
    _parsed_metrics(result)


def _require_parsed_candidate_identity(
    parsed: _ParsedComponentJob,
    plan: ExperimentPlan,
) -> None:
    expected = plan.pipeline.as_dict()
    method = expected["method"]
    configuration = expected["resolved_configuration"]
    if (
        parsed.method != method
        or parsed.configuration_name != plan.experiment_id
        or parsed.resolved_configuration != configuration
        or parsed.evidence_reference.get("seed") != plan.seed
    ):
        raise ValueError("candidate result does not match the frozen Pipeline")


def _require_parsed_recommendation_identity(
    parsed: _ParsedComponentJob,
    recommendation: PipelineRecommendation,
) -> None:
    expected = recommendation.pipeline.as_dict()
    method = expected["method"]
    configuration = expected["resolved_configuration"]
    if (
        parsed.method != method
        or parsed.configuration_name != recommendation.recommendation_id
        or parsed.resolved_configuration != configuration
        or parsed.evidence_reference.get("seed") != recommendation.seed
    ):
        raise ValueError("recommendation result does not match the frozen Pipeline")


def _require_parsed_pipeline_definition_identity(
    parsed: _ParsedComponentJob,
    plan: PipelineDefinitionExperimentPlan,
) -> None:
    expected = plan.detector_pipeline.as_dict()
    if (
        parsed.method != expected["method"]
        or parsed.configuration_name != plan.experiment_id
        or parsed.resolved_configuration != expected["resolved_configuration"]
        or parsed.evidence_reference.get("seed") != plan.seed
    ):
        raise ValueError("pipeline result does not match the frozen detector Pipeline")


def _active_evidence_baseline(
    collection: EvidenceCollection,
    scope: _ResearchScope,
    evidence_id: str,
) -> CanonicalEvidence:
    evidence = collection.by_id(evidence_id)
    if not _evidence_matches_task(evidence, scope):
        raise ValueError("baseline Evidence is outside the exact current task")
    return evidence


def _require_initial_baseline_if_needed(
    collection: EvidenceCollection,
    scope: _ResearchScope,
    baseline_evidence_id: str,
) -> None:
    if _evidence_execution_attempt_count(scope) != 0:
        return
    landscape = build_component_landscape(collection, scope.task)
    expected = landscape.default_baseline_evidence_id
    if expected is None:
        raise ValueError("initial baseline is unavailable")
    if baseline_evidence_id != expected:
        raise ValueError("first ResearchProposal baseline must use ResearchState initial baseline")


def _evidence_matches_task(
    evidence: CanonicalEvidence,
    scope: _ResearchScope,
) -> bool:
    return (
        evidence.task["dataset"] == scope.task.dataset
        and evidence.task["schema"] == scope.task.schema
        and plain_json(evidence.task["unit"]) == plain_json(scope.task.unit.as_dict())
    )


def _evidence_pipeline(evidence: CanonicalEvidence) -> PipelineSpec | NormalizedPipeline:
    payload = (
        evidence.pipeline
        if isinstance(evidence, ComponentEvidence)
        else evidence.candidate_pipeline
    )
    return resolve_baseline_pipeline(payload, str(evidence.task["dataset"]))


def _target_value_for_evidence(evidence: CanonicalEvidence, target_metric: str) -> float:
    value = evidence.metrics.get(target_metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Evidence is missing target metric: {evidence.evidence_id}")
    metric = float(value)
    if not math.isfinite(metric):
        raise ValueError(f"Evidence target metric is not finite: {evidence.evidence_id}")
    return metric


def _new_execution_result(
    *,
    scope: _ResearchScope,
    experiment_id: str,
    baseline: CanonicalEvidence,
    baseline_pipeline: Mapping[str, object],
    candidate_pipeline: Mapping[str, object],
    action: NextAction,
    hypothesis: str,
    parsed_result: _ParsedComponentResult,
    status: str,
) -> dict[str, object]:
    from .semantics import semantic_pipeline_diff

    candidate_metrics = _parsed_metrics(parsed_result)
    baseline_value = _target_value_for_evidence(baseline, scope.task.target_metric)
    candidate_value = candidate_metrics[scope.task.target_metric]
    return {
        "status": status,
        "experiment_id": experiment_id,
        "baseline_evidence_id": baseline.evidence_id,
        "hypothesis": hypothesis,
        "action": action.as_dict(),
        "baseline_pipeline": plain_json(baseline_pipeline),
        "candidate_pipeline": plain_json(candidate_pipeline),
        "semantic_diff": semantic_pipeline_diff(baseline_pipeline, candidate_pipeline, scope.task),
        "target_metric": scope.task.target_metric,
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "delta": candidate_value - baseline_value,
        "official_metrics": candidate_metrics,
        "baseline_official_metrics": dict(baseline.metrics),
        "execution_budget": _execution_budget_view(scope),
    }


def _reuse_evidence_result(
    *,
    scope: _ResearchScope,
    baseline: CanonicalEvidence,
    reused: CanonicalEvidence,
    action: NextAction,
    hypothesis: str,
) -> dict[str, object]:
    return {
        "status": "reused",
        "reused_evidence_id": reused.evidence_id,
        "baseline_evidence_id": baseline.evidence_id,
        "hypothesis": hypothesis,
        "action": action.as_dict(),
        "target_metric": scope.task.target_metric,
        "baseline_value": _target_value_for_evidence(
            baseline,
            scope.task.target_metric,
        ),
        "candidate_value": _target_value_for_evidence(
            reused,
            scope.task.target_metric,
        ),
        "delta": _target_value_for_evidence(reused, scope.task.target_metric)
        - _target_value_for_evidence(baseline, scope.task.target_metric),
        "candidate_pipeline": _evidence_pipeline(reused).normalized().canonical_payload(),
        "official_metrics": dict(reused.metrics),
        "baseline_official_metrics": dict(baseline.metrics),
        "execution_budget": _execution_budget_view(scope),
    }


def _committed_evidence_result(
    scope: _ResearchScope,
    evidence: CanonicalEvidence,
) -> dict[str, object]:
    if not isinstance(evidence, ResearchStepEvidence):
        raise ValueError("committed research result must be ResearchStepEvidence")
    from .semantics import semantic_pipeline_diff

    return {
        "recorded": True,
        "status": "committed",
        "experiment_id": evidence.evidence_id.rsplit("/", 1)[-1],
        "evidence_id": evidence.evidence_id,
        "baseline_evidence_id": evidence.baseline_evidence_id,
        "action": plain_json(evidence.action),
        "baseline_pipeline": plain_json(evidence.baseline_pipeline),
        "candidate_pipeline": plain_json(evidence.candidate_pipeline),
        "semantic_diff": semantic_pipeline_diff(evidence.baseline_pipeline, evidence.candidate_pipeline, scope.task),
        "control_note": "This is the recorded execution. Historical multi-change records are Pipeline comparisons, not single-factor causal evidence.",
        "diagnosis_status": _string(
            _mapping(evidence.research_decision, "diagnosis"),
            "verdict",
        ),
        "target_metric": evidence.target_metric,
        "baseline_value": evidence.before_value,
        "candidate_value": evidence.after_value,
        "delta": evidence.edge_delta,
        "official_metrics": dict(evidence.metrics),
        "next_action": plain_json(evidence.research_decision["next_action"]),
        "next_action_rationale": evidence.research_decision.get(
            "next_action_rationale"
        ),
        "execution_budget": _execution_budget_view(scope),
    }


def _operation_attempt(
    scope: _ResearchScope,
    experiment_id: str,
) -> Mapping[str, object] | None:
    matches = tuple(
        item
        for item in _operation_attempts(scope)
        if item.get("experiment_id") == experiment_id
    )
    if len(matches) > 1:
        raise ValueError(f"operation ledger has duplicate experiment_id: {experiment_id}")
    return matches[0] if matches else None


def _operation_attempt_with_pipeline(
    scope: _ResearchScope,
    candidate_pipeline: Mapping[str, object],
    *,
    exclude_experiment_id: str,
) -> Mapping[str, object] | None:
    key = canonical_pipeline_key_from_parts(
        task=_task_payload(scope),
        pipeline=candidate_pipeline,
    )
    matches = tuple(
        attempt
        for attempt in _operation_attempts(scope)
        if attempt.get("experiment_id") != exclude_experiment_id
        and canonical_pipeline_key_from_parts(
            task=_task_payload(scope),
            pipeline=_mapping(attempt, "candidate_pipeline"),
        )
        == key
    )
    if len(matches) > 1:
        raise ValueError("operation ledger has duplicate candidate Pipeline attempts")
    return matches[0] if matches else None


def _recover_or_reject_duplicate_pipeline_attempt(
    scope: _ResearchScope,
    attempt: Mapping[str, object],
) -> dict[str, object]:
    status = _string(attempt, "status")
    experiment_id = _string(attempt, "experiment_id")
    if status == "failed":
        raise ValueError(
            "candidate Pipeline already has a failed run-local attempt: "
            f"{experiment_id}"
        )
    return _recover_evidence_first_attempt(scope, attempt)


def _operation_attempts(scope: _ResearchScope) -> tuple[Mapping[str, object], ...]:
    payload = _read_operation_ledger(scope)
    attempts = payload.get("attempts", [])
    if not isinstance(attempts, list):
        raise TypeError("operation ledger attempts must be a list")
    return tuple(_as_mapping(item, "operation attempt") for item in attempts)


def _current_episode_feedback(
    scope: _ResearchScope,
    collection: EvidenceCollection,
) -> tuple[ResearchStepEvidence, ...]:
    feedback: list[ResearchStepEvidence] = []
    seen: set[str] = set()
    for attempt in _operation_attempts(scope):
        if attempt.get("status") != "committed":
            continue
        committed_id = attempt.get("committed_evidence_id")
        if not isinstance(committed_id, str) or not committed_id:
            raise ValueError("committed operation attempt must reference Evidence")
        if committed_id in seen:
            continue
        evidence = collection.by_id(committed_id)
        if not isinstance(evidence, ResearchStepEvidence):
            raise ValueError("episode feedback must reference ResearchStepEvidence")
        if not _evidence_matches_task(evidence, scope):
            raise ValueError("episode feedback Evidence is outside the exact current task")
        feedback.append(evidence)
        seen.add(committed_id)
    return tuple(feedback)


def _upsert_operation_attempt(
    scope: _ResearchScope,
    attempt: Mapping[str, object],
) -> None:
    _string(attempt, "experiment_id")
    attempts = list(_operation_attempts(scope))
    for index, item in enumerate(attempts):
        if item.get("experiment_id") == attempt["experiment_id"]:
            attempts[index] = plain_json(attempt)
            break
    else:
        attempts.append(plain_json(attempt))
    _write_operation_ledger(scope, {"attempts": attempts})


def _read_operation_ledger(scope: _ResearchScope) -> Mapping[str, object]:
    path = _operation_ledger_path(scope)
    if not path.exists():
        return {"attempts": []}
    return _read_mapping(path)


def _write_operation_ledger(
    scope: _ResearchScope,
    payload: Mapping[str, object],
) -> None:
    path = _operation_ledger_path(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    temporary.write_text(
        json.dumps(
            plain_json(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _operation_ledger_path(scope: _ResearchScope) -> Path:
    return scope.experiment_root / "operational-state.json"


def _evidence_execution_attempt_count(scope: _ResearchScope) -> int:
    path = _operation_ledger_path(scope)
    if not path.exists():
        return 0
    return sum(
        1
        for attempt in _operation_attempts(scope)
        if attempt.get("status") in _CONSUMED_ATTEMPT_STATUSES
    )


def _pipeline_action_attempt_count(scope: _ResearchScope) -> int:
    if not scope.experiment_root.exists():
        return 0
    return sum(
        1
        for path in scope.experiment_root.glob("*/pipeline-trajectory.json")
        if path.is_file()
    )


def _execution_budget_view(scope: _ResearchScope) -> dict[str, object]:
    used = _evidence_execution_attempt_count(scope)
    budget = {
        "experiments_used": used,
        "max_experiments": scope.max_experiments,
        "experiments_remaining": max(0, scope.max_experiments - used),
    }
    attempts = _operation_attempts(scope)
    budget["branch_runs_reserved"] = sum(
        int(item.get("branch_runs", 1)) for item in attempts
        if item.get("status") in _CONSUMED_ATTEMPT_STATUSES
    )
    budget["max_branch_runs"] = 2 * scope.max_experiments
    budget["branch_runs_remaining"] = max(0, budget["max_branch_runs"] - budget["branch_runs_reserved"])
    return budget


def _task_payload(scope: _ResearchScope) -> dict[str, object]:
    return {
        "dataset": scope.task.dataset,
        "schema": scope.task.schema,
        "unit": scope.task.unit.as_dict(),
    }


def _optional_action_parameter(action: Mapping[str, object]) -> str | None:
    parameter = action.get("parameter")
    if parameter is None:
        return None
    if not isinstance(parameter, str):
        raise TypeError("action.parameter must be a string or null")
    return parameter


def _research_artifact_from_reference(
    artifact: Mapping[str, object],
    experiment_dir: Path,
) -> tuple[dict[str, object], dict[str, object] | None, tuple[Path, str] | None]:
    artifact_type = _string(artifact, "artifact_type")
    selection_status = _string(artifact, "selection_status")
    score_source = artifact.get("score_source")
    validate_score_lineage(selection_status, artifact_type, score_source)  # type: ignore[arg-type]
    state = artifact.get("state")
    artifact_name: str | None = None
    artifact_source: tuple[Path, str] | None = None
    if state is not None:
        if not isinstance(state, str):
            raise TypeError("Model Artifact state reference must be a string or null")
        artifact_name = {
            "torch_state_dict": "best.pt",
            "fitted_estimator": "estimator.pkl",
        }.get(artifact_type)
        if artifact_name is None:
            raise ValueError(f"unexpected artifact state for {artifact_type}")
        source = experiment_dir / state
        if not source.is_file():
            raise FileNotFoundError(f"Model Artifact state is missing: {source}")
        artifact_source = (source, artifact_name)
    summary = {
        "artifact_type": artifact_type,
        "selection_status": selection_status,
        "score_source": score_source,
        "artifact_path": artifact_name,
    }
    metadata: dict[str, object] | None = None
    metadata_ref = artifact.get("metadata")
    if metadata_ref is not None and artifact_type != "replay_recipe":
        if not isinstance(metadata_ref, str):
            raise TypeError("Model Artifact metadata reference must be a string")
        metadata = _artifact_metadata_for_agent_memory(
            _read_mapping(experiment_dir / metadata_ref),
            artifact_name,
        )
    return summary, metadata, artifact_source


def _parsed_metrics(result: _ParsedComponentResult) -> dict[str, float]:
    metrics = {}
    for metric in OFFICIAL_METRIC_IDS:
        try:
            value = result.metrics[metric]
        except KeyError as exc:
            raise ValueError(f"Component result is missing official metric: {metric}") from exc
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Component result metric is not numeric: {metric}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Component result metric is not finite: {metric}")
        metrics[metric] = numeric
    return metrics


def _commit_research_step_evidence(
    *,
    scope: _ResearchScope,
    evidence: ResearchStepEvidence,
    artifact_metadata: Mapping[str, object] | None,
    artifact_source: tuple[Path, str] | None,
    pipeline_bundle: Path | None = None,
) -> str:
    final_dir = evidence_directory(scope.agent_memory_root, evidence)
    final_path = final_dir / "evidence.json"
    if final_path.exists():
        existing = ResearchStepEvidence.from_dict(_read_mapping(final_path))
        if existing != evidence:
            raise ValueError(f"ResearchStepEvidence already exists: {evidence.evidence_id}")
        if pipeline_bundle is not None:
            from .pipeline_execution import validate_pipeline_bundle

            validate_pipeline_bundle(final_dir, evidence.candidate_pipeline)
        return evidence.evidence_id

    staging_root = scope.agent_memory_root / ".staging" / uuid4().hex
    staging_root.mkdir(parents=True, exist_ok=False)
    try:
        if pipeline_bundle is not None:
            from .pipeline_execution import bundle_file, validate_pipeline_bundle

            binding = validate_pipeline_bundle(pipeline_bundle, evidence.candidate_pipeline)
            for name in (*binding["files"], "pipeline-artifact.json", "run.json"):
                _link_or_copy(bundle_file(pipeline_bundle, name), staging_root / name)
        if artifact_metadata is not None:
            _write_json(staging_root / "artifact.json", artifact_metadata)
        if artifact_source is not None:
            source, name = artifact_source
            _link_or_copy(source, staging_root / name)
        _write_json(staging_root / "evidence.json", evidence.as_dict())
        ResearchStepEvidence.from_dict(_read_mapping(staging_root / "evidence.json"))
        validate_evidence_artifact(scope.agent_memory_root, evidence, artifact_directory=staging_root)
        if pipeline_bundle is not None:
            validate_pipeline_bundle(staging_root, evidence.candidate_pipeline)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_root.rename(final_dir)
    except Exception:
        raise
    return evidence.evidence_id


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _require_research_decision_matches(
    evidence: CanonicalEvidence,
    payload: Mapping[str, object],
) -> None:
    if not isinstance(evidence, ResearchStepEvidence):
        raise ValueError("committed research result must be ResearchStepEvidence")
    expected = {
        "diagnosis": {
            "verdict": _string(payload, "diagnosis_status"),
            "reason": _string(payload, "diagnosis"),
        },
        "next_action": research_next_action_from_mapping(
            payload,
            "next_action",
        ).as_dict(),
        "next_action_rationale": _string(payload, "next_action_rationale"),
    }
    decision = evidence.research_decision
    actual = {
        "diagnosis": plain_json(decision["diagnosis"]),
        "next_action": plain_json(decision["next_action"]),
        "next_action_rationale": decision.get("next_action_rationale"),
    }
    if actual != expected:
        raise ValueError("ResearchStepEvidence already has conflicting research facts")


def _run_experiment(
    scope: _ResearchScope,
    payload: Mapping[str, object],
) -> dict[str, object]:
    has_action = "action" in payload
    has_pipeline_action = "pipeline_action" in payload
    if has_action == has_pipeline_action:
        raise ValueError("experiment.run requires exactly one of action or pipeline_action")
    if has_pipeline_action:
        return _run_pipeline_action_experiment(scope, payload)
    return _run_evidence_first_experiment(scope, payload)


def _run_pipeline_action_experiment(
    scope: _ResearchScope,
    payload: Mapping[str, object],
) -> dict[str, object]:
    collection = EvidenceCollection.from_directory(scope.agent_memory_root)
    baseline = _active_evidence_baseline(collection, scope, _string(payload, "baseline_evidence_id"))
    action = _requested_action(scope, payload, baseline)
    translated = dict(payload)
    del translated["pipeline_action"]
    translated["action"] = action.as_dict()
    return _run_evidence_first_experiment(scope, translated)


def _freeze_recommendation(
    scope: _ResearchScope,
    payload: Mapping[str, object],
) -> dict[str, object]:
    recommendation = _recommendation_from_payload(scope, payload)
    recommendation_dir = scope.experiment_root / recommendation.recommendation_id
    recommendation_path = recommendation_dir / "recommendation.json"
    frozen = recommendation.as_dict()
    if recommendation_path.exists():
        if _read_mapping(recommendation_path) != frozen:
            raise ValueError("recommendation.json contains conflicting facts")
    elif recommendation_dir.exists() and any(recommendation_dir.iterdir()):
        raise ValueError("recommendation output already contains evidence")
    else:
        _write_json(recommendation_path, frozen)
    return {
        "recorded": True,
        "recommendation_id": recommendation.recommendation_id,
        "recommendation_job": str(recommendation_dir.resolve()),
        "action": deepcopy(dict(recommendation.action)),
        "pipeline": recommendation.pipeline.as_dict(),
        "rationale": recommendation.rationale,
    }


def _evaluate_recommendation(
    scope: _ResearchScope,
    payload: Mapping[str, object],
) -> dict[str, object]:
    recommendation_id = _recommendation_id(payload)
    recommendation_dir = scope.experiment_root / recommendation_id
    evaluation_path = recommendation_dir / "evaluation.json"
    recommendation = _recommendation_from_evidence(scope, recommendation_dir)
    if evaluation_path.exists():
        return dict(_read_mapping(evaluation_path))

    if (recommendation_dir / "run.json").exists():
        result_root = recommendation_dir
    else:
        outcome = run_recommendation_evaluation(
            recommendation,
            dataset_root=scope.dataset_root,
            dataset_source_root=scope.dataset_source_root,
            output_dir=recommendation_dir,
            zero_shot_membership=scope.zero_shot_membership,
        )
        result_root = outcome.job_dir
    parsed = parse_validated_component_results(
        result_root,
        scope.dataset_dir,
        {
            "source": scope.dataset_source,
            "revision": scope.dataset_revision,
        },
        profile_dir=scope.profile_dir,
    )
    result = _parsed_result_for_unit(parsed, scope)
    _require_parsed_scope(parsed, result, scope)
    _require_parsed_recommendation_identity(parsed, recommendation)
    metrics = _parsed_metrics(result)
    evaluation = {
        "recommendation_id": recommendation.recommendation_id,
        "recommendation_job": str(recommendation_dir.resolve()),
        "task": scope.task.as_dict(),
        "target_metric": scope.task.target_metric,
        "target_value": metrics[scope.task.target_metric],
        "official_metrics": metrics,
        "result_case_count": 1,
        "pipeline": recommendation.pipeline.as_dict(),
        "rationale": recommendation.rationale,
    }
    _write_json(evaluation_path, evaluation)
    return evaluation


def _recommendation_from_payload(
    scope: _ResearchScope,
    payload: Mapping[str, object],
) -> PipelineRecommendation:
    recommendation_id = _recommendation_id(payload)
    method = _string(payload, "method")
    action = _recommendation_action(payload)
    return PipelineRecommendation(
        recommendation_id=recommendation_id,
        task=scope.task,
        rationale=_string(payload, "rationale"),
        action=action,
        seed=1,
        budget={"max_runs": 1, "max_wall_seconds": scope.max_wall_seconds},
        pipeline=_recommendation_pipeline(method, action, scope.task),
    )


def _recommendation_from_evidence(
    scope: _ResearchScope,
    recommendation_dir: Path,
) -> PipelineRecommendation:
    payload = _read_mapping(recommendation_dir / "recommendation.json")
    task = TaskContext.from_dict(_mapping(payload, "task"))
    if task != scope.task:
        raise ValueError("Recommendation task does not match the research scope")
    budget = _mapping(payload, "budget")
    expected_budget = {
        "max_runs": 1,
        "max_wall_seconds": scope.max_wall_seconds,
    }
    if budget != expected_budget:
        raise ValueError("Recommendation budget does not match the frozen scope")
    pipeline_payload = _mapping(payload, "pipeline")
    method = _string(pipeline_payload, "method")
    action = _recommendation_action(payload)
    pipeline = _recommendation_pipeline(method, action, task)
    recommendation = PipelineRecommendation(
        recommendation_id=_string(payload, "recommendation_id"),
        task=task,
        rationale=_string(payload, "rationale"),
        action=action,
        seed=_positive_integer(payload, "seed"),
        budget=budget,
        pipeline=pipeline,
    )
    if recommendation.as_dict() != payload:
        raise ValueError("Recommendation Evidence does not match its frozen plan")
    return recommendation


def _recommendation_pipeline(
    method: str,
    action: Mapping[str, object],
    task: TaskContext,
) -> PipelineSpec:
    try:
        spec = METHOD_CATALOG[method]
    except KeyError as exc:
        raise ValueError(f"unknown Catalog method: {method}") from exc
    capability = spec.capability(task.schema)
    if not capability.supported:
        raise ValueError(
            f"{method} is N/A for {task.schema}: {capability.unavailable_reason}"
        )
    baseline = PipelineSpec.official_default(method, task.dataset)
    if action["dimension"] == "default":
        return baseline
    return candidate_pipeline_from_action(
        baseline,
        NextAction(
            dimension=str(action["dimension"]),  # type: ignore[arg-type]
            parameter=(
                str(action["parameter"])
                if action.get("parameter") is not None
                else None
            ),
            value=deepcopy(action.get("value")),
        ),
        task,
    )


def _recommendation_action(payload: Mapping[str, object]) -> dict[str, object]:
    action = _mapping(payload, "action")
    _exact_fields(action, {"dimension", "parameter", "value"}, "action")
    if _string(action, "dimension") == "default":
        if action.get("parameter") is not None:
            raise ValueError("default recommendation action requires null parameter")
        if action.get("value") != "official_default":
            raise ValueError("default recommendation action requires official_default")
        return {
            "dimension": "default",
            "parameter": None,
            "value": "official_default",
        }
    if action.get("dimension") == "algorithm":
        raise ValueError("recommendation method already selects the algorithm")
    if action.get("dimension") in {"pipeline", "preprocessing", "ensemble"}:
        raise ValueError("complete Pipeline planning uses experiment.run; recommendation remains native-only")
    return _action(payload, "action").as_dict()


def _recommendation_id(payload: Mapping[str, object]) -> str:
    value = _string(payload, "recommendation_id")
    if _EXPERIMENT_ID.fullmatch(value) is None:
        raise ValueError("recommendation_id contains unsupported path characters")
    return value


def _complete_research_step(
    scope: _ResearchScope,
    payload: Mapping[str, object],
) -> dict[str, object]:
    if _operation_attempt(scope, _experiment_id(payload)) is None:
        raise ValueError("ResearchStep completion requires a pending execution attempt")
    return _complete_evidence_first_research_step(scope, payload)


def _finish_episode(
    scope: _ResearchScope,
    payload: Mapping[str, object],
) -> dict[str, object]:
    if any(item.get("status") == "succeeded_pending_diagnosis" for item in _operation_attempts(scope)):
        raise ValueError("complete the pending diagnosis before finishing the episode")
    reason = _string(payload, "reason")
    if reason not in _FINISH_EPISODE_REASONS:
        raise ValueError(f"unsupported episode finish reason: {reason}")
    return {
        "status": "finished",
        "finish_reason": reason,
        "rationale": _string(payload, "rationale"),
        "execution_budget": _execution_budget_view(scope),
    }


def _recommendation_catalog_view(scope: _ResearchScope) -> list[dict[str, object]]:
    catalog = []
    for method in sorted(METHOD_CATALOG):
        spec = METHOD_CATALOG[method]
        capability = spec.capability(scope.task.schema)
        if not capability.supported:
            continue
        configuration = resolve_configuration(
            spec.module,
            (scope.task.dataset,),
        )[scope.task.dataset]
        model_params = configuration.get("Model_Params")
        window_parameter = spec.sample_spec.window_parameter
        catalog.append(
            {
                "method": method,
                "fit_scope": capability.fit_scope,
                "default_pipeline": {
                    "method": method,
                    "resolved_configuration": deepcopy(configuration),
                },
                "allowed_actions": {
                    "default": "official_default",
                    "preprocess": ["raw", "min-max", "z-score"],
                    "diff_order": {"minimum": 0},
                    "window": (
                        None
                        if window_parameter is None
                        else {"parameter": window_parameter}
                    ),
                    "hyperparameters": (
                        sorted(
                            str(key)
                            for key in model_params
                            if str(key) != window_parameter
                        )
                        if isinstance(model_params, Mapping)
                        else []
                    ),
                },
                "parameter_semantics": {
                    "window_parameter": spec.sample_spec.window_parameter,
                    "step_parameter": spec.sample_spec.step_parameter,
                    "horizon": spec.sample_spec.horizon,
                },
            }
        )
    return catalog

def _action(payload: Mapping[str, object], key: str) -> NextAction:
    action = _mapping(payload, key)
    _exact_fields(action, {"dimension", "parameter", "value"}, key)
    dimension = _string(action, "dimension")
    parameter = action.get("parameter")
    if parameter is not None and not isinstance(parameter, str):
        raise TypeError(f"{key}.parameter must be a string or null")
    return NextAction(
        dimension=dimension,  # type: ignore[arg-type]
        parameter=parameter,
        value=deepcopy(action.get("value")),
    )


def _pipeline_action(payload: Mapping[str, object], key: str) -> PipelineAction:
    return PipelineAction.from_dict(_mapping(payload, key))


def _experiment_id(payload: Mapping[str, object]) -> str:
    value = _string(payload, "experiment_id")
    if _EXPERIMENT_ID.fullmatch(value) is None:
        raise ValueError("experiment_id contains unsupported path characters")
    return value


def _read_mapping(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"research evidence is missing: {path}") from None
    return _as_mapping(payload, str(path))


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _exact_fields(
    payload: Mapping[str, object],
    expected: set[str],
    contract: str,
    *,
    optional: set[str] | None = None,
) -> None:
    optional_fields = set() if optional is None else optional
    allowed = expected | optional_fields
    unknown = sorted(set(payload) - allowed)
    missing = sorted(expected - set(payload))
    if unknown:
        raise ValueError(f"Unknown {contract} field: {unknown[0]}")
    if missing:
        raise ValueError(f"Missing {contract} field: {missing[0]}")


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _as_mapping(payload.get(key), key)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _positive_integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


__all__ = ["dispatch_research_tool"]
