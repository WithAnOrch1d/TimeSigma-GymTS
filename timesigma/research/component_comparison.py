"""Complete read-only Component tables for explicitly selected source tasks.

The supplied collection is the visibility boundary. Never reopen canonical
Evidence from disk to fill holes in a frozen or masked collection.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from timesigma.algorithms import METHOD_CATALOG

from .canonical import ComponentEvidence
from .collection import plain_json, read_json_mapping
from .memory_access import _require_development, evidence_visible
from .provider_context import decision_semantics
from .retrieval import _ProfileCache, _profile_vector, _similarity, PROFILE_VECTOR_FIELDS


def _names(values, limit, label):
    if (not isinstance(values, (list, tuple)) or not 1 <= len(values) <= limit
            or any(not isinstance(x, str) or not x.strip() for x in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{label} requires 1–{limit} distinct names")


def _pipeline_differences(reference, candidate, path=()):
    """Exact field differences, not an intervention or causal classification."""
    changes = []
    for key in sorted(reference.keys() | candidate.keys()):
        before, after = reference.get(key), candidate.get(key)
        field = [*path, key]
        if key in reference and key in candidate and isinstance(before, Mapping) and isinstance(after, Mapping):
            changes.extend(_pipeline_differences(before, after, field))
        elif key not in reference or key not in candidate or before != after:
            changes.append({"path": field,
                            "reference": {"present": key in reference, "value": before},
                            "candidate": {"present": key in candidate, "value": after}})
    return changes


def _distance(profiles, source, target_vector):
    source_vector = profiles.vector_for_task(source)
    present = bool(target_vector and source_vector)
    similarity, distance = _similarity(target_vector, source_vector) if present else (None, None)
    return {
        "status": "available" if present else "profile_unavailable",
        "value": distance, "similarity": similarity,
        "source_profile": source_vector,
        "shared_fields": [f for f in PROFILE_VECTOR_FIELDS if f in target_vector and f in (source_vector or {})],
        "target_only_fields": [f for f in PROFILE_VECTOR_FIELDS if f in target_vector and f not in (source_vector or {})],
        "source_only_fields": [f for f in PROFILE_VECTOR_FIELDS if f in (source_vector or {}) and f not in target_vector],
    }


def compare_component_sources(
    collection, task, memory_root: Path, *, evidence_ids, candidate_methods,
    metric, one_shot, target_profile,
):
    _names(evidence_ids, 4, "source anchors")
    _names(candidate_methods, 12, "candidate_methods")
    if not one_shot:
        _require_development(task)
    for method in candidate_methods:
        spec = METHOD_CATALOG.get(method)
        if (spec is None or not spec.agent_available or not spec.supports_dataset(task.dataset)
                or not spec.capability(task.schema).supported):
            raise ValueError(f"candidate is not available in the target Catalog scope: {method}")

    manifest = read_json_mapping(memory_root / "manifest.json")
    anchors = []
    for identifier in evidence_ids:
        record = collection.by_id(identifier)
        if not evidence_visible(record, task, manifest, one_shot=one_shot):
            raise ValueError("Evidence is outside the visible research scope")
        if any(record.task == other.task for other in anchors):
            raise ValueError("use exactly one Evidence anchor per source task")
        anchors.append(record)

    profiles = _ProfileCache(memory_root.parent / "profiles", manifest)
    # Reuse stored Profiles; do not fit detectors, read labels or materialize new
    # Profiles. Delivery/isolated callers can supply their allowed train view.
    target_vector = _profile_vector(target_profile) if target_profile is not None else (
        profiles.vector_for_task(task.as_dict()) if task.dataset.casefold() != "aiops" else None
    )
    target_vector = target_vector or {}
    sources, rows = [], []
    for index, anchor in enumerate(anchors):
        source = plain_json(anchor.task)
        sources.append({"source_index": index, "anchor_evidence_id": anchor.evidence_id,
                        "source_task": source, "source_distance": _distance(profiles, source, target_vector)})
        # Same complete unit (including ordered members), same supplied view.
        observed = {
            record.pipeline["method"]: record for record in collection.evidence
            if isinstance(record, ComponentEvidence) and record.task == anchor.task
            and evidence_visible(record, task, manifest, one_shot=one_shot)
        }
        for method in candidate_methods:
            spec = METHOD_CATALOG[method]
            source_supported = spec.supports_dataset(source["dataset"]) and spec.capability(source["schema"]).supported
            record = observed.get(method)
            # N/A and absent are different. Never backfill with another Schema,
            # a Research variant, a different unit, or a zero score.
            status = "observed" if record is not None else "missing" if source_supported else "source_unsupported"
            rows.append({
                "source_index": index, "method": method, "status": status,
                "source_schema_supported": source_supported,
                "evidence_id": record.evidence_id if record else None,
                "candidate_pipeline": plain_json(record.pipeline) if record else None,
                "metric": metric, "value": record.metrics[metric] if record else None,
                "artifact": {k: v for k, v in record.artifact.items() if k != "artifact_path"} if record else None,
                "configuration_comparison": None,
            })

    # Neutral reference = first present candidate in caller order, never best
    # score. Full identities remain in every observed cell for lossless reading.
    for source in sources:
        present = [row for row in rows if row["source_index"] == source["source_index"] and row["status"] == "observed"]
        if present:
            reference = present[0]
            for row in present:
                row["configuration_comparison"] = {
                    "reference_evidence_id": reference["evidence_id"],
                    "differences": _pipeline_differences(reference["candidate_pipeline"], row["candidate_pipeline"]),
                }
    across_sources = []
    for method in candidate_methods:
        present = [row for row in rows if row["method"] == method and row["status"] == "observed"]
        for row in present[1:]:
            across_sources.append({
                "method": method, "reference_evidence_id": present[0]["evidence_id"],
                "evidence_id": row["evidence_id"],
                "differences": _pipeline_differences(present[0]["candidate_pipeline"], row["candidate_pipeline"]),
            })
    semantics = decision_semantics(task)
    return {
        "source": "caller-frozen source tasks × candidate Methods; canonical Components in the supplied visible collection",
        "metric": metric, "candidate_methods": list(candidate_methods), "source_tasks": sources,
        "records": rows,
        "coverage": {"expected_cells": len(anchors) * len(candidate_methods),
                     **{status: sum(row["status"] == status for row in rows)
                        for status in ("observed", "missing", "source_unsupported")},
                     "note": "Complete grid for this request and collection, not completeness of the global store. Missing is unknown, never zero or inferior."},
        "cross_source_configuration_differences": across_sources,
        "configuration_note": "References are first present records in caller order, not recommendations. Compare entire recorded Pipelines (including operators/ensemble), not Method names alone. Dataset-conditioned defaults may differ; no claim of a controlled intervention or equivalence to current defaults.",
        "target_profile": target_vector,
        "distance_definition": "Existing retrieval: signed-log1p field values, RMS difference, one-sided missing field penalty 1, similarity=1/(1+distance). Train members only for shared units. Not a calibrated transfer probability; no cutoff or source reselection.",
        "artifact_semantics": {k: semantics[k] for k in ("eligibility", "artifact_selection", "new_fit_vs_load")},
        "artifact_note": "Recorded metadata only; no checkpoint load or fresh artifact validation. selection_status is not Method eligibility; score_source is historical lineage, not load compatibility.",
        "store_protocol": {"seed": manifest.get("seed"), "easytsad_revision": manifest.get("easytsad_revision"),
                           "dataset_revisions": {s["source_task"]["dataset"]: manifest.get("dataset_revisions", {}).get(s["source_task"]["dataset"]) for s in sources}},
        "comparison_note": "No ranking, aggregate winner, automatic decision, Research substitution, or target outcome. Inspect contradictions across sources; observed performance differences do not establish transfer or causal effects.",
    }
