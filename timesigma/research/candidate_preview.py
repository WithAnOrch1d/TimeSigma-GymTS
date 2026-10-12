from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any, Mapping
from uuid import uuid4

from .collection import read_json_mapping, write_json_mapping_if_same
from .pipeline import NextAction, TaskContext, apply_next_action, normalize_pipeline, validate_declared_pipeline_change
from .semantics import semantic_pipeline_diff


def preview_candidate(baseline: Any, action: NextAction, task: TaskContext) -> dict[str, Any]:
    result = {
        "baseline_pipeline": normalize_pipeline(baseline, task.dataset).canonical_payload(),
        "requested_action": action.as_dict(),
        "candidate_pipeline": None,
        "semantic_diff": None,
    }
    try:
        candidate = apply_next_action(baseline, action, task)
        result["candidate_pipeline"] = candidate.normalized().canonical_payload()
        result["semantic_diff"] = semantic_pipeline_diff(baseline, candidate, task)
        result["canonical_preprocessing"] = {
            "baseline": normalize_pipeline(baseline, task.dataset).preprocessing_spec(task.schema).as_dict(),
            "candidate": candidate.normalized().preprocessing_spec(task.schema).as_dict(),
        }
        validate_declared_pipeline_change(baseline, action, candidate, task)
    except (ValueError, TypeError, KeyError) as error:
        return {**result, "status": "incompatible", "reason": str(error), "detector_runs": 0}
    return {**result, "status": "ready", "detector_runs": 2 if candidate.normalized().ensemble_ref else 1}


def save_preview(root: Path, binding: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    preview_id = uuid4().hex
    write_json_mapping_if_same(root / "previews" / (preview_id + ".json"), {"binding": dict(binding), "candidate_pipeline": dict(candidate)})
    return preview_id


def check_preview(root: Path, preview_id: Any, binding: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    if not isinstance(preview_id, str) or re.fullmatch(r"[0-9a-f]{32}", preview_id) is None:
        return False
    path = root / "previews" / (preview_id + ".json")
    if not path.is_file():
        return False
    saved = read_json_mapping(path)
    return json.dumps(saved, sort_keys=True) == json.dumps({"binding": dict(binding), "candidate_pipeline": dict(candidate)}, sort_keys=True)
