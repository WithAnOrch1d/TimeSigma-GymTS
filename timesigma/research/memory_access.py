from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from .canonical import ComponentEvidence, ResearchStepEvidence, planning_unit_identity
from .collection import CanonicalEvidence, EvidenceCollection, evidence_directory, plain_json, read_json_mapping
from .pipeline import PlanningUnit, TaskContext
from .provider_context import _pipeline_summary, decision_semantics
from .retrieval import _ProfileCache, _profile_vector, _similarity
from .semantics import context_window_parameter, semantic_pipeline_diff


MEMORY_TOOLS = ("memory.search", "memory.read", "memory.compare", "capability.lookup")
SEARCH_FIELDS = (
    "kind", "task_scope", "dataset", "schema", "planning_unit", "detector",
    "preprocessing", "diff_order", "window", "parameter", "parameter_value",
    "component", "ensemble", "action_dimension", "delta_sign", "diagnosis", "metric",
)


def search_memory(
    collection: EvidenceCollection,
    task: TaskContext,
    memory_root: Path,
    profile: Mapping[str, Any],
    *,
    query: str,
    filters: Sequence[Mapping[str, Any]],
    order: str,
    offset: int,
    limit: int,
    one_shot: bool = False,
) -> dict[str, Any]:
    if not one_shot:
        _require_development(task)
    manifest = read_json_mapping(memory_root / "manifest.json")
    _page(offset, limit)
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if order not in {"evidence_id", "profile_similarity"}:
        raise ValueError("order must be evidence_id or profile_similarity")
    selected_filters = _filters(filters)
    profiles = _ProfileCache(memory_root.parent / "profiles", manifest)
    target_vector = _profile_vector(profile)
    matches = []
    for evidence in collection.evidence:
        if not evidence_visible(evidence, task, manifest, one_shot=one_shot):
            continue
        if not _matches(evidence, task, selected_filters):
            continue
        searchable = {
            "summary": evidence_summary(evidence),
            "pipeline": plain_json(_pipeline(evidence)),
            "reasoning": plain_json(evidence.research_decision) if isinstance(evidence, ResearchStepEvidence) else None,
        }
        text = json.dumps(searchable, ensure_ascii=False).casefold()
        if not all(term in text for term in query.casefold().split()):
            continue
        similarity = None
        if order == "profile_similarity":
            vector = profiles.vector_for_task(evidence.task)
            if not target_vector or not vector:
                continue
            similarity, _ = _similarity(target_vector, vector)
        matches.append((evidence, similarity))
    matches.sort(key=lambda item: (
        -(item[1] or 0.0) if order == "profile_similarity" else 0,
        item[0].evidence_id,
    ))
    items = []
    for evidence, similarity in matches[offset:offset + limit]:
        summary = evidence_summary(evidence)
        if similarity is not None:
            summary["profile_similarity"] = similarity
        items.append(summary)
    return {
        "source": "canonical Evidence",
        "query": query,
        "filters": dict(selected_filters),
        "order": order,
        "ordering_note": "Requested search ordering, not a performance recommendation.",
        "total_matches": len(matches),
        "offset": offset,
        "next_offset": offset + limit if offset + limit < len(matches) else None,
        "items": items,
    }


def read_memory(
    collection: EvidenceCollection,
    task: TaskContext,
    memory_root: Path,
    *,
    evidence_ids: Sequence[str],
    include_lineage: bool,
    offset: int,
    limit: int,
    one_shot: bool = False,
) -> dict[str, Any]:
    if not one_shot:
        _require_development(task)
    _page(offset, limit)
    if not isinstance(evidence_ids, (list, tuple)) or not 1 <= len(evidence_ids) <= 3:
        raise ValueError("read requires one to three Evidence IDs")
    if not all(isinstance(item, str) and item for item in evidence_ids):
        raise TypeError("Evidence IDs must be non-empty strings")
    if not isinstance(include_lineage, bool):
        raise TypeError("include_lineage must be boolean")
    manifest = read_json_mapping(memory_root / "manifest.json")
    profiles = _ProfileCache(memory_root.parent / "profiles", manifest)
    items = []
    for evidence_id in evidence_ids:
        evidence = collection.by_id(evidence_id)
        if not evidence_visible(evidence, task, manifest, one_shot=one_shot):
            raise ValueError("Evidence is outside the visible research scope")
        step = isinstance(evidence, ResearchStepEvidence)
        artifact = plain_json(evidence.artifact)
        reference = artifact.pop("artifact_path")
        directory = evidence_directory(memory_root, evidence)
        artifact["saved_state_present"] = reference is not None and (directory / reference).is_file()
        artifact["pipeline_bundle_present"] = (directory / "pipeline-artifact.json").is_file()
        item = {
            "evidence_id": evidence.evidence_id,
            "kind": evidence.kind,
            "source_task": _task_summary(evidence.task),
            "source_profile": profiles.vector_for_task(evidence.task),
            "profile_note": "Train-side curve statistics; shared units average the member Profiles.",
            "baseline_evidence_id": evidence.baseline_evidence_id if step else None,
            "baseline_pipeline": plain_json(evidence.baseline_pipeline) if step else None,
            "candidate_pipeline": plain_json(_pipeline(evidence)),
            "hypothesis": evidence.research_decision["hypothesis"] if step else None,
            "action": plain_json(evidence.action) if step else None,
            "target_metric": evidence.target_metric if step else None,
            "official_metrics": dict(evidence.metrics),
            "before_value": evidence.before_value if step else None,
            "after_value": evidence.after_value if step else None,
            "delta": evidence.edge_delta if step else None,
            "research_decision": plain_json(evidence.research_decision) if step else None,
            "diagnosis_note": "Historical diagnosis is the recorded model interpretation, not an established law.",
            "semantic_diff": _recorded_semantics(evidence),
            "control_note": "Historical action labels are not proof of single-factor control. Interpret multi-change records only as comparisons of the actual resolved Pipelines; do not expand their causal claims.",
            "canonical_preprocessing": {
                "candidate": evidence.canonical_view()["candidate_pipeline" if step else "pipeline"]["preprocessing"],
                "baseline": evidence.canonical_view()["baseline_pipeline"]["preprocessing"] if step else None,
            },
            "artifact": artifact,
        }
        if include_lineage:
            parent_id = evidence.baseline_evidence_id if step else None
            parent = collection.by_id(parent_id) if parent_id else None
            relatives = []
            for relative in collection.evidence:
                if not isinstance(relative, ResearchStepEvidence) or relative.evidence_id == evidence_id:
                    continue
                relation = (
                    "child" if relative.baseline_evidence_id == evidence_id
                    else "sibling" if parent_id and relative.baseline_evidence_id == parent_id
                    else None
                )
                if relation and evidence_visible(relative, task, manifest, one_shot=one_shot):
                    relatives.append({"relation": relation, **evidence_summary(relative)})
            relatives.sort(key=lambda relative: (relative["relation"], relative["evidence_id"]))
            item["branch_context"] = {
                "parent": evidence_summary(parent) if parent is not None and evidence_visible(parent, task, manifest, one_shot=one_shot) else None,
                "items": relatives[offset:offset + limit],
                "total_matches": len(relatives),
                "next_offset": offset + limit if offset + limit < len(relatives) else None,
                "note": "baseline_evidence_id is the scientific parent; Ensemble branches stay inside the Pipeline.",
            }
        items.append(item)
    return {"source": "canonical Evidence", "items": items,
            "artifact_note": decision_semantics(task)["eligibility"]}


def evidence_visible(
    evidence: CanonicalEvidence,
    target: TaskContext,
    manifest: Mapping[str, Any],
    *,
    one_shot: bool = False,
) -> bool:
    if str(evidence.task["dataset"]).casefold() == "aiops":
        return False
    if target.dataset.casefold() == "aiops":
        return one_shot
    if _same_task(evidence.task, target):
        return not one_shot
    target_payload = {"dataset": target.dataset, "unit": target.unit.as_dict()}
    source_members = _source_members(evidence.task, manifest)
    target_members = _source_members(target_payload, manifest)
    return bool(source_members and target_members) and source_members.isdisjoint(target_members)


def evidence_summary(evidence: CanonicalEvidence) -> dict[str, Any]:
    step = isinstance(evidence, ResearchStepEvidence)
    diagnosis = evidence.research_decision["diagnosis"] if step else None
    semantic = _recorded_semantics(evidence)
    return {
        "evidence_id": evidence.evidence_id,
        "kind": evidence.kind,
        "source_task": _task_summary(evidence.task),
        "baseline_evidence_id": evidence.baseline_evidence_id if step else None,
        "baseline_summary": _short_pipeline(evidence.baseline_pipeline) if step else None,
        "candidate_summary": _short_pipeline(_pipeline(evidence)),
        "action": _short_action(evidence.action) if step else None,
        "target_metric": evidence.target_metric if step else None,
        "delta": evidence.edge_delta if step else None,
        "diagnosis": {"verdict": diagnosis["verdict"], "reason": str(diagnosis["reason"])[:300]} if diagnosis else None,
        "semantic_dimensions": [item["semantic_dimension"] for item in semantic["changes"]] if semantic else [],
        "single_semantic_change": semantic["single_semantic_change"] if semantic else None,
    }


def _recorded_semantics(evidence: CanonicalEvidence) -> dict[str, Any] | None:
    if not isinstance(evidence, ResearchStepEvidence):
        return None
    source = evidence.task
    task = TaskContext(str(source["dataset"]), str(source["schema"]), evidence.target_metric, PlanningUnit.from_dict(str(source["schema"]), source["unit"]))
    return semantic_pipeline_diff(evidence.baseline_pipeline, evidence.candidate_pipeline, task)


def _filters(filters: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(filters, (list, tuple)) or len(filters) > len(SEARCH_FIELDS):
        raise ValueError("filters must be a bounded array")
    selected = {}
    for item in filters:
        if not isinstance(item, Mapping) or set(item) != {"field", "value"}:
            raise ValueError("each filter requires field and value")
        field, value = item["field"], item["value"]
        if not isinstance(field, str) or field not in SEARCH_FIELDS or field in selected:
            raise ValueError("unknown or duplicate Memory filter")
        if field not in {"diff_order", "window", "parameter_value"} and not isinstance(value, str):
            raise TypeError(f"{field} filter requires a string")
        if field in {"diff_order", "window"} and (isinstance(value, bool) or not isinstance(value, int)):
            raise TypeError(f"{field} filter requires an integer")
        choices = {
            "kind": {"component", "research_step"},
            "task_scope": {"current", "other", "all"},
            "delta_sign": {"positive", "zero", "negative"},
            "ensemble": {"none", "mean"},
        }
        if field in choices and value not in choices[field]:
            raise ValueError(f"invalid {field} filter")
        selected[field] = value
    if "parameter_value" in selected and "parameter" not in selected:
        raise ValueError("parameter_value requires a parameter filter")
    return selected


def _matches(evidence: CanonicalEvidence, target: TaskContext, filters: Mapping[str, Any]) -> bool:
    pipeline = _pipeline(evidence)
    detectors = [pipeline]
    ensemble = pipeline.get("ensemble_ref")
    if ensemble:
        detectors.extend(ensemble["detectors"])
    data = pipeline["resolved_configuration"]["Data_Params"]
    operators = pipeline.get("operators", ())
    components = {item["component_id"] for item in operators}
    if data["diff_order"]:
        components.update(("differencing", "native_diff_order"))
    if data["preprocess"] != "raw":
        components.update(("scaling", "native_preprocess", data["preprocess"]))
    if "robust_scaler" in components:
        components.update(("scaling", "robust"))
    if not components:
        components.add("none")
    components.add(data["preprocess"])
    if ensemble:
        components.update(("ensemble", "mean_score_fusion", "mean"))
    step = isinstance(evidence, ResearchStepEvidence)
    for field, value in filters.items():
        if field == "kind" and value != evidence.kind:
            return False
        if field == "task_scope" and value != "all" and (_same_task(evidence.task, target) != (value == "current")):
            return False
        if field in {"dataset", "schema"} and value != evidence.task[field]:
            return False
        if field == "planning_unit" and value != planning_unit_identity(str(evidence.task["schema"]), evidence.task["unit"]):
            return False
        if field == "detector" and not any(value == item["method"] for item in detectors):
            return False
        if field in {"component", "preprocessing"} and value not in components:
            return False
        if field == "diff_order" and value != data["diff_order"]:
            return False
        if field == "ensemble" and value != ("mean" if ensemble else "none"):
            return False
        if field == "action_dimension" and (not step or value != evidence.action["dimension"]):
            return False
        if field == "delta_sign":
            if not step or value != ("positive" if evidence.edge_delta > 0 else "negative" if evidence.edge_delta < 0 else "zero"):
                return False
        if field == "diagnosis" and (not step or value.casefold() not in json.dumps(plain_json(evidence.research_decision["diagnosis"])).casefold()):
            return False
        if field == "metric" and (value not in evidence.metrics or (step and value != evidence.target_metric)):
            return False
        if field in {"window", "parameter"}:
            parameters = [item["resolved_configuration"]["Model_Params"] for item in detectors]
            parameters.extend(item.get("parameters", {}) for item in operators)
            if field == "window":
                names = {context_window_parameter(item["method"]) for item in detectors} | {"window"}
                if not any(params.get(name) == value for params in parameters for name in names if name is not None):
                    return False
            elif not any(value in params and ("parameter_value" not in filters or plain_json(params[value]) == filters["parameter_value"]) for params in parameters):
                return False
    return True


def _pipeline(evidence: CanonicalEvidence) -> Mapping[str, Any]:
    return evidence.pipeline if isinstance(evidence, ComponentEvidence) else evidence.candidate_pipeline


def _short_pipeline(pipeline: Mapping[str, Any]) -> dict[str, Any]:
    summary = _pipeline_summary(pipeline)
    summary.pop("model_params")
    reference = pipeline.get("ensemble_ref")
    summary.pop("ensemble_ref", None)
    summary["detectors"] = [pipeline["method"], *([item["method"] for item in reference["detectors"]] if reference else [])]
    return summary


def _short_action(action: Mapping[str, Any]) -> dict[str, Any]:
    payload = plain_json(action)
    if payload["dimension"] == "ensemble":
        payload = {"dimension": "ensemble", "before": "none" if action["before"] is None else "mean", "after": "none" if action["after"] is None else "mean"}
    return payload


def _task_summary(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": task["dataset"], "schema": task["schema"],
        "planning_unit": planning_unit_identity(str(task["schema"]), task["unit"]),
        "training_member_count": len(task["unit"]["training_curves"]),
        "held_out_member_count": len(task["unit"]["held_out_curves"]),
    }


def _same_task(source: Mapping[str, Any], target: TaskContext) -> bool:
    return source["dataset"] == target.dataset and source["schema"] == target.schema and plain_json(source["unit"]) == target.unit.as_dict()


def _source_members(task: Mapping[str, Any], manifest: Mapping[str, Any]) -> set[tuple[str, str, str, str]]:
    dataset = str(task["dataset"])
    source = manifest.get("dataset_sources", {}).get(dataset, manifest.get("dataset_source"))
    revision = manifest.get("dataset_revisions", {}).get(dataset)
    if source is None or revision is None:
        return set()
    unit = task["unit"]
    return {(source, revision, dataset, curve) for curve in (*unit["training_curves"], *unit["held_out_curves"])}


def _page(offset: int, limit: int) -> None:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 8:
        raise ValueError("limit must be between one and eight")


def _require_development(task: TaskContext) -> None:
    if task.dataset.casefold() == "aiops":
        raise ValueError("AIOPS cannot enter research Memory tools")
