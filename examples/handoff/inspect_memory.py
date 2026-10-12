"""Read the real handoff sample through existing TimeSigma Memory interfaces.

No detector execution, Provider request, score evaluation or Memory write occurs.
Run from the repository root with PYTHONPATH=.; see the adjacent README.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from timesigma.research.agent_memory import validate_agent_memory_store
from timesigma.research.collection import EvidenceCollection
from timesigma.research.evidence_facts import compare_evidence
from timesigma.research.memory_access import read_memory, search_memory
from timesigma.research.pipeline import PlanningUnit, TaskContext


ROOT = Path(__file__).resolve().parent


def verify_sample() -> dict:
    receipt = json.loads((ROOT / "sample-source.json").read_text())
    for row in receipt["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid sample receipt path")
        data = (ROOT / "sample" / relative).read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"sample bytes changed: {relative}")
    return receipt


def main() -> None:
    receipt = verify_sample()
    memory = ROOT / "sample/agent-memory"
    collection = EvidenceCollection.from_directory(memory)
    validation = validate_agent_memory_store(memory, load_artifacts=True).as_dict()
    if (validation["component_evidence"] != 4 or validation["research_step_evidence"] != 2
            or validation["aiops_evidence"] != 0 or not validation["manifest_ok"]
            or validation["missing_artifacts"] or validation["absolute_path_violations"]):
        raise ValueError("handoff sample failed structural/Artifact validation")

    task = TaskContext(
        dataset="TODS", schema="naive", target_metric="point-wise-f1",
        unit=PlanningUnit.for_schema(
            "naive", curve="0", training_curves=("0",), held_out_curves=(),
        ),
    )
    manifest = json.loads((memory / "manifest.json").read_text())
    profile_path = ROOT / "sample/profiles/TODS" / manifest["dataset_revisions"]["TODS"] / "curves/0.json"
    profile = {"curve_profile": json.loads(profile_path.read_text())["facts"]}
    search = search_memory(
        collection, task, memory, profile, query="", filters=[],
        order="profile_similarity", offset=0, limit=8,
    )
    if search["total_matches"] != 6:
        raise ValueError("sample Evidence/Profile search is incomplete")
    comparison = compare_evidence(
        collection, task, memory,
        evidence_ids=["component/AR/naive/TODS/234"],
        candidate_methods=["AR", "OCSVM", "LOF"],
        metric="point-wise-f1", target_profile=profile,
    )
    # LOF is deliberately absent from this sample, not assigned a poor score.
    rows = {row["method"]: row for row in comparison["records"]}
    if (rows["LOF"]["status"] != "missing" or rows["LOF"]["value"] is not None
            or any(rows[method]["status"] != "observed" for method in ("AR", "OCSVM"))):
        raise ValueError("sample comparison lost explicit missingness")
    details = read_memory(
        collection, task, memory,
        evidence_ids=[
            "research-step/TODS/naive/234/tod234_exp1_ar_p32",
            "research-step/TODS/naive/234/tod234_exp2_ar_p8",
        ],
        include_lineage=True, offset=0, limit=8,
    )
    verify_sample()
    print(json.dumps({
        "status": "PASS", "purpose": receipt["purpose"],
        "validation": validation, "search": search,
        "comparison": comparison, "research_steps": details,
        "limits": "Read/lineage/Artifact loading only; no inference replay, training, LLM or full Research run.",
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
