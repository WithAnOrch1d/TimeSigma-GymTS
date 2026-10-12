"""Read-only arithmetic and claim checks over visible canonical Evidence.

These views neither rank target Pipelines nor certify transfer hypotheses.
They use the existing store visibility and identity, never result directories.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
import math
from pathlib import Path
from typing import Any

from timesigma.benchmark.leaderboard import OFFICIAL_METRIC_IDS

from .canonical import ResearchStepEvidence
from .collection import EvidenceCollection, plain_json, read_json_mapping
from .memory_access import _pipeline, _recorded_semantics, _require_development, evidence_visible
from .pipeline import TaskContext
from .provider_context import lookup_capabilities


def _visible_records(collection, task, memory_root, ids, *, one_shot):
    if not one_shot:
        _require_development(task)
    manifest = read_json_mapping(memory_root / "manifest.json")
    records = {}
    for identifier in ids:
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("full Evidence IDs are required")
        evidence = collection.by_id(identifier)
        if not evidence_visible(evidence, task, manifest, one_shot=one_shot):
            raise ValueError("Evidence is outside the visible research scope")
        records[identifier] = evidence
    return records, manifest


def _metric(metric):
    if not isinstance(metric, str) or metric not in OFFICIAL_METRIC_IDS:
        raise ValueError("metric must be an official EasyTSAD metric ID")


def _relation(left, right):
    return "gt" if left > right else "lt" if left < right else "eq"


def compare_evidence(
    collection: EvidenceCollection, task: TaskContext, memory_root: Path, *,
    evidence_ids: Sequence[str], metric: str, one_shot: bool = False,
    candidate_methods: Sequence[str] | None = None,
    target_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare records, or expand caller-chosen source units into a Component table.

    In table mode evidence_ids are source anchors, not a candidate shortlist.
    target_profile is Runner-owned input, never an Agent tool argument.
    """
    _metric(metric)
    if candidate_methods is not None and not isinstance(candidate_methods, (list, tuple)):
        raise ValueError("candidate_methods must be an array")
    if candidate_methods:
        from .component_comparison import compare_component_sources

        return compare_component_sources(
            collection, task, memory_root, evidence_ids=evidence_ids,
            candidate_methods=candidate_methods, metric=metric,
            one_shot=one_shot, target_profile=target_profile,
        )
    if not isinstance(evidence_ids, (list, tuple)) or not 2 <= len(evidence_ids) <= 12:
        raise ValueError("compare requires 2–12 distinct Evidence IDs")
    if any(not isinstance(x, str) for x in evidence_ids) or len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("compare requires distinct full Evidence IDs")
    records, manifest = _visible_records(collection, task, memory_root, evidence_ids, one_shot=one_shot)
    rows, groups = [], []
    for evidence in records.values():
        pipeline = plain_json(_pipeline(evidence))
        source = plain_json(evidence.task)
        row = {
            "evidence_id": evidence.evidence_id, "kind": evidence.kind,
            "source_task": source, "candidate_pipeline": pipeline,
            "metric": metric, "value": evidence.metrics[metric],
            "artifact": {k: v for k, v in evidence.artifact.items() if k != "artifact_path"},
            "comparison_scope": "complete recorded configuration; not a controlled intervention",
        }
        if isinstance(evidence, ResearchStepEvidence):
            row.update({
                "comparison_scope": "recorded Research transition; interpret its actual semantic diff",
                "baseline_evidence_id": evidence.baseline_evidence_id,
                "baseline_pipeline": plain_json(evidence.baseline_pipeline),
                "semantic_diff": _recorded_semantics(evidence),
                # A transition delta belongs only to its declared metric.
                "transition": {"metric": evidence.target_metric, "before_value": evidence.before_value,
                               "after_value": evidence.after_value, "delta": evidence.edge_delta},
            })
        rows.append(row)
        group = next((item for item in groups if item["source_task"] == source), None)
        if group is None:
            group = {"source_task": source, "evidence_ids": [], "present_methods": []}
            groups.append(group)
        group["evidence_ids"].append(evidence.evidence_id)
        if pipeline["method"] not in group["present_methods"]:
            group["present_methods"].append(pipeline["method"])
    methods = sorted({_pipeline(e)["method"] for e in records.values()})
    pairs = []
    for group in groups:
        group["methods_not_in_requested_evidence"] = [m for m in methods if m not in group["present_methods"]]
        for left_id, right_id in combinations(group["evidence_ids"], 2):
            left, right = records[left_id].metrics[metric], records[right_id].metrics[metric]
            pairs.append({"left_id": left_id, "right_id": right_id,
                          "left_minus_right": left - right, "relation": _relation(left, right)})
    return {
        "source": "canonical Evidence; caller-selected records, no target ranking",
        "metric": metric, "records": rows, "same_task_pairs": pairs,
        "coverage": {"methods": methods, "source_tasks": groups,
                     "tasks_present_by_method": {m: sum(m in g["present_methods"] for g in groups) for m in methods},
                     "note": "Coverage describes this request only. Missing is unknown, never zero or inferior. Method coverage does not merge different configurations."},
        "store_protocol": {"seed": manifest.get("seed"), "easytsad_revision": manifest.get("easytsad_revision"),
                           "dataset_revisions": {g["source_task"]["dataset"]: manifest.get("dataset_revisions", {}).get(g["source_task"]["dataset"]) for g in groups}},
        "comparison_note": "Pairs require identical dataset, Schema and full member lists within this canonical store. Differences describe observed scores, not causal effects or transfer confidence. No cross-task subtraction or winner recommendation.",
    }


def verify_fact_claims(
    collection: EvidenceCollection, task: TaskContext, memory_root: Path, *,
    claims: Sequence[Mapping[str, Any]], one_shot: bool = False,
) -> dict[str, Any]:
    """Check submitted facts, not natural-language rationales or scientific predictions."""
    if not isinstance(claims, (list, tuple)) or len(claims) > 24:
        raise ValueError("fact_claims must be an array of at most 24 structured claims")
    ids = []
    fields = {
        "metric": {"kind", "evidence_id", "metric", "value"},
        "comparison": {"kind", "left_id", "right_id", "metric", "relation"},
        "capability": {"kind", "method", "available_in_current_schema"},
    }
    for claim in claims:
        if not isinstance(claim, Mapping) or not isinstance(claim.get("kind"), str) or claim["kind"] not in fields or set(claim) != fields[claim["kind"]]:
            raise ValueError("unknown or malformed structured fact claim")
        kind = claim["kind"]
        if kind != "capability":
            _metric(claim["metric"])
            ids.extend([claim["evidence_id"]] if kind == "metric" else [claim["left_id"], claim["right_id"]])
        if kind == "metric" and (isinstance(claim["value"], bool) or not isinstance(claim["value"], (float, int)) or not math.isfinite(claim["value"])):
            raise ValueError("claimed metric must be a finite number")
        if kind == "comparison" and claim["relation"] not in ("gt", "eq", "lt"):
            raise ValueError("comparison relation must be gt, eq or lt")
        if kind == "capability" and (not isinstance(claim["method"], str) or not claim["method"].strip() or not isinstance(claim["available_in_current_schema"], bool)):
            raise ValueError("capability claims require a Method name and boolean availability")
    records, _ = _visible_records(collection, task, memory_root, ids, one_shot=one_shot)
    checks = []
    for claim in claims:
        kind = claim["kind"]
        if kind == "metric":
            actual = records[claim["evidence_id"]].metrics[claim["metric"]]
            correct = math.isclose(actual, claim["value"], rel_tol=0, abs_tol=1e-12)
            note = "Recorded metric value; 1e-12 arithmetic precision, not a replay tolerance."
        elif kind == "comparison":
            left, right = records[claim["left_id"]], records[claim["right_id"]]
            same_task = left.task == right.task
            actual = _relation(left.metrics[claim["metric"]], right.metrics[claim["metric"]]) if same_task else None
            correct = same_task and actual == claim["relation"]
            note = "Same source task and metric; observed order only." if same_task else "Different source tasks cannot establish a same-task ranking."
        else:
            cards = lookup_capabilities(task, kind="detector", name=claim["method"], one_shot=one_shot)["items"]
            actual = cards[0]["available_in_current_schema"] if cards else None
            correct = actual is not None and actual == claim["available_in_current_schema"]
            note = "Catalog Schema eligibility, independent of historical Artifact selection status; native feasibility remains separately checked."
        checks.append({"claim": dict(claim), "correct": correct, "observed": actual, "note": note})
    return {"status": "no_claims" if not checks else "pass" if all(c["correct"] for c in checks) else "fail",
            "checks": checks, "evidence_ids": list(dict.fromkeys(ids)),
            "limits": "Only submitted structured facts were checked. No verification of free text, completeness of claims, causal explanation, transfer prediction or expected target performance."}
