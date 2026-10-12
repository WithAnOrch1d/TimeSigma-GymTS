from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from timesigma.profiles.dataset import build_dataset_profiles
from timesigma.profiles.features import CURVE_PROFILE_DEFINITION, profile_curve
from timesigma.benchmark.zero_shot import MEMBERSHIP_FILENAME
from timesigma.research.collection import read_json_mapping, write_json_mapping_if_same
from timesigma.research.memory_access import evidence_visible, read_memory, search_memory
from timesigma.research.evidence_facts import compare_evidence, verify_fact_claims
from timesigma.research.pipeline import PipelineSpec
from timesigma.research.pipeline_execution import file_digest
from timesigma.research.planning import pipeline_detectors, plan_pipeline
from timesigma.research.provider_context import decision_semantics, lookup_capabilities
from timesigma.research.semantics import context_window_parameter, execution_only_parameters

from .protocol import frozen_memory, load_protocol, planning_units, task_for


def dispatch_planning(tool: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if tool == "memory.read":
        ids = payload.get("evidence_ids")
        if not isinstance(ids, (list, tuple)) or not 1 <= len(ids) <= 3:
            raise ValueError("memory.read accepts 1–3 Evidence IDs per call; split a longer list into separate calls")
    root = Path(payload["delivery_root"]).resolve()
    protocol = load_protocol(root)
    curve = payload["curve"]
    task = task_for(protocol, curve)
    if (root / "planning-closed.json").exists() or (root / "scores-frozen.json").exists():
        raise ValueError("planning is closed; no target results may return to an Agent")
    unit = root / "plans" / curve
    if (unit / "final.json").exists():
        raise ValueError("this target already has its one frozen legal plan")
    start_path = unit / "planning-start.json"
    now = datetime.now(timezone.utc).timestamp()
    if not start_path.exists():
        if tool != "delivery.context":
            raise ValueError("load delivery context first")
        write_json_mapping_if_same(start_path, {"started_unix": now, "deadline_unix": now + protocol["planning"]["max_wall_seconds"]})
    if now >= read_json_mapping(start_path)["deadline_unix"]:
        raise ValueError("frozen planning wall-time exhausted")
    profile_path = unit / "profile.json"
    if not profile_path.exists():
        dataset_dir = Path(protocol["dataset_root"]) / "UTS" / task.dataset
        if task.schema == "naive":
            train_path = dataset_dir / curve / "train.npy"
            facts = {"curve_profile": profile_curve(np.load(train_path, allow_pickle=False)).as_dict(), "train_sha256": file_digest(train_path)}
        else:
            aggregate, = build_dataset_profiles(
                dataset=task.dataset, dataset_dir=dataset_dir, schema=task.schema,
                provenance=protocol["dataset_provenance"],
                zero_shot_membership=root / MEMBERSHIP_FILENAME if task.schema == "zero_shot" else None,
            )
            facts = {**aggregate.as_dict(), "train_sha256": {
                member: file_digest(dataset_dir / member / "train.npy")
                for member in task.unit.training_curves
            }}
            if task.schema == "zero_shot":
                facts["membership_sha256"] = protocol["zero_shot_membership_sha256"]
        write_json_mapping_if_same(profile_path, {
            **facts, "task": task.as_dict(), "input_state": "raw_train",
            "profile_definition": CURVE_PROFILE_DEFINITION,
            "dataset_revision": protocol["dataset_revision"],
            "persistence": "run-local protocol artifact; never persistent Research Memory",
        })
    profile = dict(read_json_mapping(profile_path))
    if tool == "delivery.context":
        snapshot = read_json_mapping(root / protocol["memory_snapshot"])
        return {
            "task": task.as_dict(), "profile": profile,
            "decision_semantics": decision_semantics(task),
            **({"zero_shot_provenance": protocol["zero_shot_provenance"]} if task.schema == "zero_shot" else {}),
            "initial_baseline": None, "target_results": "unavailable and forbidden",
            "objective": protocol["objective"], "budgets": {k: protocol[k] for k in ("planning", "execution")},
            "memory": {
                "mode": "on", "read_only": True, "snapshot": protocol["memory_snapshot"],
                "target_results_allowed": False,
                "historical_datasets": sorted({item[0] for item in snapshot["datasets_schemas"]}),
                "historical_datasets_by_schema": {
                    schema: sorted({dataset for dataset, recorded_schema in snapshot["datasets_schemas"] if recorded_schema == schema})
                    for schema in sorted({item[1] for item in snapshot["datasets_schemas"]})
                },
                "excluded_datasets": ["AIOPS"],
                "visibility": "Only frozen non-target Evidence; same-target and overlapping-source members across Schemas are excluded. AIOPS history is intentionally unavailable; query reformulation cannot expose it.",
            },
            "capabilities": {
                "detectors": [card["name"] for card in protocol["capability_snapshot"]["items"] if card["kind"] == "detector" and card["available_in_current_schema"]],
                "preprocessing": [card["name"] for card in protocol["capability_snapshot"]["items"] if card["kind"] == "preprocessing"],
                "ensemble": "none or existing two-detector mean; query capability_lookup for exact parameters and compatibility",
            },
            "contract": "One complete initial plan, no target baseline or research transition. All target plans close before any detector executes; all scores freeze before evaluation.",
        }
    if tool == "capability.lookup":
        return lookup_capabilities(task, kind=payload["kind"], name=payload["name"], one_shot=True)
    if tool in {"memory.search", "memory.read", "memory.compare"}:
        if tool == "memory.search" and (
            re.search(r"\baiops\b", str(payload.get("query", "")), flags=re.IGNORECASE)
            or any(item.get("field") == "dataset" and str(item.get("value")).casefold() == "aiops" for item in payload.get("filters", ()))
        ):
            return {"status": "protocol_excluded", "dataset": "AIOPS", "items": [], "next_offset": None,
                    "reason": "AIOPS historical Evidence is intentionally unavailable. Reformulating an AIOPS-history query cannot expose hidden records; use the eligible non-target datasets in delivery_context."}
        collection = frozen_memory(root)
        memory_root = Path(protocol["memory_root"])
        request = {k: v for k, v in payload.items() if k not in {"delivery_root", "curve"}}
        if tool == "memory.compare":
            return compare_evidence(collection, task, memory_root, one_shot=True, target_profile=profile, **request)
        if tool == "memory.search":
            result = search_memory(collection, task, memory_root, profile, one_shot=True, **request)
            # Component summaries have no research transition; omit null placeholders
            # from this delivery view while retaining every recorded value and ID.
            result["items"] = [{k: v for k, v in item.items() if v is not None} for item in result["items"]]
            return result
        return read_memory(collection, task, memory_root, one_shot=True, **request)
    if tool == "delivery.preview":
        return _preview(root, unit, protocol, task, payload["pipeline"], payload.get("fact_claims", []))
    if tool == "delivery.freeze":
        preview = read_json_mapping(unit / "preview.json")
        if payload["preview_id"] != preview["preview_id"]:
            raise ValueError("freeze requires this target's resolved preview")
        rationale = payload["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("planning rationale is required")
        ids = payload["evidence_ids"]
        if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
            raise ValueError("Evidence references must be a list of full IDs")
        collection = frozen_memory(root)
        manifest = read_json_mapping(Path(protocol["memory_root"]) / "manifest.json")
        for evidence_id in ids:
            if not evidence_visible(collection.by_id(evidence_id), task, manifest, one_shot=True):
                raise ValueError("plan cites target or invisible Evidence")
        checked_ids = preview.get("fact_verification", {}).get("evidence_ids", [])
        if not set(checked_ids).issubset(ids):
            raise ValueError("freeze must retain Evidence IDs supporting the checked facts")
        plan = {
            "task": task.as_dict(), "profile_reference": str(profile_path.relative_to(root)),
            "profile_sha256": file_digest(profile_path), "memory_snapshot_reference": protocol["memory_snapshot"],
            "memory_snapshot_sha256": protocol["memory_snapshot_sha256"], "agent": protocol["provider"],
            **dict(preview), "planning_rationale": rationale, "evidence_ids": list(dict.fromkeys(ids)),
            "status": "frozen", "source_revision": protocol["source_revision"],
            "protocol_sha256": file_digest(root / "protocol.json"),
        }
        write_json_mapping_if_same(unit / "final.json", plan)
        return {"status": "frozen", "curve": curve, "plan_reference": str((unit / "final.json").relative_to(root))}
    raise ValueError("tool is not part of one-shot planning")


def _preview(root, unit, protocol, task, proposal, fact_claims):
    ready = unit / "preview.json"
    if ready.exists():
        previous = read_json_mapping(ready)
        if previous["proposal"] == proposal and previous.get("fact_claims", []) == fact_claims:
            return dict(previous)
        raise ValueError("one legal candidate is already resolved; cannot sample another legal plan")
    attempts = sorted((unit / "invalid-plans").glob("*.json"))
    if len(attempts) >= protocol["planning"]["max_invalid_plans"]:
        raise ValueError("invalid-plan repair budget exhausted")
    # A changed Memory snapshot is an integrity failure, not an Agent-repairable
    # claim or Pipeline error. Keep it outside the bounded invalid-plan handler.
    collection = frozen_memory(root)
    verification = None
    try:
        verification = verify_fact_claims(
            collection, task, Path(protocol["memory_root"]),
            claims=fact_claims, one_shot=True,
        )
        if verification["status"] == "fail":
            raise ValueError("structured facts contradict visible Evidence or Catalog; correct them before locking a Pipeline")
        pipeline = plan_pipeline(proposal, task)
        initializations = []
        for detector in pipeline_detectors(pipeline):
            defaults = PipelineSpec.official_default(detector.method, task.dataset)
            models = dict(detector.resolved_configuration["Model_Params"])
            for name in execution_only_parameters():
                if models.get(name) != defaults.resolved_configuration["Model_Params"].get(name):
                    raise ValueError("execution-only defaults cannot be changed by the Agent")
            private = {k: v for k, v in models.items() if k not in {context_window_parameter(detector.method), "step"}}
            initializations.append({"method": detector.method, "resolved_private_parameters": private, "rule": "explicit Agent overrides, otherwise pinned Method defaults; not independent findings"})
        branches = len(pipeline_detectors(pipeline))
        if branches > protocol["execution"]["max_branches_per_plan"]:
            raise ValueError("plan exceeds frozen detector branch limit")
    except (ValueError, TypeError, KeyError) as error:
        failure = {"status": "invalid", "reason": str(error), "proposal": proposal,
                   "fact_claims": fact_claims, "fact_verification": verification}
        write_json_mapping_if_same(unit / "invalid-plans" / f"{len(attempts)+1}.json", failure)
        return {**failure, "repairs_remaining": protocol["planning"]["max_invalid_plans"] - len(attempts) - 1}
    from uuid import uuid4

    preview = {
        "status": "ready", "preview_id": uuid4().hex, "proposal": proposal,
        "fact_claims": fact_claims, "fact_verification": verification,
        "pipeline": pipeline.canonical_payload(), "private_initialization": initializations,
        "canonical_pipeline": pipeline.canonical_view(task.schema),
        "compatibility": {"valid": True, "schema": task.schema, "scope": "declared Pipeline compatibility; native execution checks data/runtime feasibility"},
        "expected_detector_branches": branches,
        "semantic_context": "initial complete plan; no baseline transition or target experiment",
    }
    write_json_mapping_if_same(ready, preview)
    return preview


def close_planning(root: Path) -> dict[str, Any]:
    protocol = load_protocol(root)
    frozen_memory(root)
    plans = {}
    for curve in planning_units(protocol):
        path = root / "plans" / curve / "final.json"
        plan = read_json_mapping(path)
        usage = read_json_mapping(path.parent / "provider-outcome.json")
        if plan["status"] != "frozen" or usage["status"] != "frozen":
            raise ValueError("every target requires a completed planning session")
        if usage["usage"]["providerAttempts"] > protocol["planning"]["max_calls"] or usage["usage"]["tokens"] > protocol["planning"]["max_tokens"]:
            raise ValueError("planning exceeded its frozen Provider budget")
        if plan["protocol_sha256"] != file_digest(root / "protocol.json"):
            raise ValueError("plan and protocol binding mismatch")
        if plan["profile_sha256"] != file_digest(root / plan["profile_reference"]):
            raise ValueError("allowed Profile changed after plan freeze")
        plans[str(path.relative_to(root))] = file_digest(path)
    record = {"protocol_sha256": file_digest(root / "protocol.json"), "plans": plans, "status": "planning_closed"}
    write_json_mapping_if_same(root / "planning-closed.json", record)
    return record
