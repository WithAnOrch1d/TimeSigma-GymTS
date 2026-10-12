from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import filecmp
from importlib.metadata import distribution, version
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from timesigma.algorithms import EASYTSAD_REVISION, EASYTSAD_SCHEMAS, METHOD_CATALOG
from timesigma.benchmark.manifest import dataset_provenance
from timesigma.benchmark.zero_shot import MEMBERSHIP_FILENAME, freeze_zero_shot_membership
from timesigma.research.canonical import ComponentEvidence, ResearchStepEvidence
from timesigma.research.collection import EvidenceCollection, evidence_directory, read_json_mapping, write_json_mapping_if_same
from timesigma.research.pipeline import PipelineSpec, PlanningUnit, TaskContext
from timesigma.research.pipeline_execution import file_digest
from timesigma.research.provider_context import lookup_capabilities

REPOSITORY = Path(__file__).resolve().parents[2]


def planning_units(protocol: dict[str, Any]) -> list[str]:
    return protocol["curves"] if protocol["schema"] == "naive" else [protocol["dataset"]]


def evaluation_curves(protocol: dict[str, Any]) -> list[str]:
    return protocol["zero_shot_split"]["held_out_curves"] if protocol["schema"] == "zero_shot" else protocol["curves"]


def task_for(protocol: dict[str, Any], curve: str) -> TaskContext:
    if curve not in planning_units(protocol):
        raise ValueError("planning unit is outside the frozen target list")
    schema = protocol["schema"]
    split = protocol.get("zero_shot_split", {})
    training = (curve,) if schema == "naive" else tuple(split.get("training_curves", protocol["curves"]))
    return TaskContext(protocol["dataset"], schema, protocol["primary_metric"], PlanningUnit.for_schema(
        schema, curve=curve if schema == "naive" else None,
        training_curves=training, held_out_curves=tuple(split.get("held_out_curves", ())),
    ))


def source_files() -> list[Path]:
    return sorted([
        *REPOSITORY.glob("timesigma/**/*.py"), *REPOSITORY.glob("pi-host/src/*.ts"),
        *REPOSITORY.glob("pi-host/dist/*.js"),
        *(REPOSITORY / name for name in (
            "pyproject.toml", "uv.lock", "pi-host/package.json", "pi-host/package-lock.json",
        )),
    ])


def supplemental_baselines(requested: object, defaults: dict, baselines: list[dict], snapshot: dict) -> list[dict]:
    """Select only explicitly frozen, supported methods absent from the public comparison."""
    if not isinstance(requested, list) or any(not isinstance(m, str) for m in requested) or len(requested) != len(set(requested)):
        raise ValueError("supplemental_baselines must contain distinct Method names")
    if not requested:
        return []
    public_methods = {row["method"] for row in snapshot["rows"]}
    if any(method not in defaults or method in public_methods for method in requested):
        raise ValueError("supplemental baseline must be supported Core and absent from frozen public snapshot")
    return [row for row in baselines if row["name"] in requested]


def prepare_protocol(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("delivery output must be new; never replace a previous target attempt")
    config = dict(read_json_mapping(config_path))
    comparison = config.get("comparison")
    if comparison is not None:
        if comparison["kind"] != "published_leaderboard_snapshot":
            raise ValueError("unknown frozen comparison mode")
        snapshot_path = Path(comparison["snapshot"])
        if file_digest(snapshot_path) != comparison["snapshot_sha256"]:
            raise ValueError("public comparison snapshot changed before prepare")
        public_snapshot = read_json_mapping(snapshot_path)
        if public_snapshot["dataset"] != config["dataset"] or public_snapshot["schema"].lower() != config["schema"]:
            raise ValueError("public snapshot dataset/schema differs from this delivery")
    if config["schema"] not in EASYTSAD_SCHEMAS or config["primary_metric"] != "point-wise-f1":
        raise ValueError("first-demo requires an official Schema and point-wise-f1")
    dataset_root = Path(config["dataset_root"]).resolve()
    provenance = dataset_provenance(Path(config["dataset_source_root"]))
    if provenance["clean"] is not True or provenance["revision"] != config["dataset_revision"]:
        raise ValueError("dataset must match its clean frozen revision")
    names = sorted(p.name for p in (dataset_root / "UTS" / config["dataset"]).iterdir() if p.is_dir())
    curves = config["curves"]
    if not curves or curves != sorted(set(curves)) or set(curves) - set(names):
        raise ValueError("target list must contain unique sorted existing members")
    if config["dataset"].casefold() == "aiops" and curves != names:
        raise ValueError("AIOPS requires the complete curve list")
    if config["schema"] != "naive" and curves != names:
        raise ValueError("shared Schemas require the complete dataset member list")
    for group in ("planning", "execution", "evaluation"):
        for name, value in config[group].items():
            if group == "execution" and name == "gpu_memory_mb" and value is None:
                continue  # GPU capacity is infrastructure; null retains normal allocator behavior.
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("budgets must be frozen positive integers")
    memory_root = Path(config["memory_root"]).resolve()
    collection = EvidenceCollection.from_directory(memory_root)
    if any(str(item.task["dataset"]).casefold() == "aiops" for item in collection.evidence):
        raise ValueError("AIOPS Evidence is forbidden in the frozen Memory")
    manifest = read_json_mapping(memory_root / "manifest.json")
    if manifest["easytsad_revision"] != EASYTSAD_REVISION:
        raise ValueError("Memory and Method source revisions differ")
    evidence_files = [evidence_directory(memory_root, item) / "evidence.json" for item in collection.evidence]
    knowledge_files = [memory_root / "manifest.json", memory_root / "zero-shot-membership.json"]
    for dataset, revision in manifest["dataset_revisions"].items():
        if dataset.casefold() == "aiops":
            raise ValueError("AIOPS cannot enter Memory Profiles")
        knowledge_files.extend(sorted((memory_root.parent / "profiles" / dataset / revision).rglob("*.json")))
    snapshot = {
        "cutoff_utc": datetime.now(timezone.utc).isoformat(),
        "authority": str(memory_root),
        "manifest": dict(manifest),
        "evidence_ids": sorted(item.evidence_id for item in collection.evidence),
        "evidence_files": {str(p): file_digest(p) for p in evidence_files},
        "knowledge_files": {str(p): file_digest(p) for p in knowledge_files},
        "counts": dict(Counter(item.kind for item in collection.evidence)),
        "datasets_schemas": sorted({(str(item.task["dataset"]), str(item.task["schema"])) for item in collection.evidence}),
        "aiops_evidence_count": 0,
        "excluded_sources": ["AIOPS", "runs/history", "separate pilot experiment views", "uncommitted diagnoses", "post-cutoff Evidence"],
        "policy": "Read-only cutoff reference to the existing authority; verify exact bytes before use. No target Artifact loading or Memory writes.",
    }
    recovery = Path(config["plan_recovery_source"]).resolve() if config.get("plan_recovery_source") else None
    if config.get("memory_snapshot_source"):
        previous_snapshot = dict(read_json_mapping(Path(config["memory_snapshot_source"])))
        if Path(previous_snapshot["authority"]).resolve() != memory_root:
            raise ValueError("frozen Memory reference must resolve to this execution copy of the authority")
        frozen = _read_frozen_memory(previous_snapshot)
        if sorted(item.evidence_id for item in frozen.evidence) != snapshot["evidence_ids"]:
            raise ValueError("resource qualification must preserve the exact Memory cutoff")
        if previous_snapshot["manifest"] != snapshot["manifest"]:
            raise ValueError("resource qualification cannot replace Memory sources")
        snapshot = previous_snapshot
    if recovery is not None:
        previous = load_protocol(recovery, verify_source=False)
        if config["dataset"].casefold() == "aiops" or previous["dataset"].casefold() == "aiops":
            raise ValueError("automatic source-repair continuation is only for non-AIOPS engineering dry-runs")
        frozen_memory(recovery)
        if (recovery / "scores-frozen.json").exists() or list(recovery.glob("**/Evals/**/*.json")):
            raise ValueError("dry-run source repair cannot use any evaluation feedback")
        old_snapshot = dict(read_json_mapping(recovery / "memory-snapshot.json"))
        for key in ("evidence_ids", "evidence_files", "knowledge_files", "manifest"):
            if old_snapshot[key] != snapshot[key]:
                raise ValueError("dry-run recovery must retain identical frozen knowledge")
        snapshot = old_snapshot
    root.mkdir(parents=True, exist_ok=True)
    write_json_mapping_if_same(root / "memory-snapshot.json", snapshot)
    defaults = {}
    baselines = []
    for method, spec in sorted(METHOD_CATALOG.items()):
        if not spec.agent_available or not spec.capability(config["schema"]).supported or not spec.supports_dataset(config["dataset"]):
            continue
        defaults[method] = PipelineSpec.official_default(method, config["dataset"]).normalized().canonical_payload()
        baselines.append({"name": method, "method": method, "model_artifacts": True, "lifecycle": "matched_catalog_selection"})
        if spec.artifact_type == "torch_state_dict":
            baselines.append({"name": method + "_upstream", "method": method, "model_artifacts": False, "lifecycle": "upstream_default"})
    protocol = {
        **config, "dataset_root": str(dataset_root), "memory_root": str(memory_root),
        "dataset_provenance": provenance, "baselines": baselines, "default_pipelines": defaults,
        "easytsad_revision": EASYTSAD_REVISION,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
        "source_state": "HEAD plus exact frozen working-tree source copies in source/; includes uncommitted implementation",
        "memory_snapshot": "memory-snapshot.json", "memory_snapshot_sha256": file_digest(root / "memory-snapshot.json"),
        "status": "prepared", "objective": "optimization_leaderboard", "variant": "TimeSigma_First_Demo",
    }
    if comparison is not None:
        # An explicit frozen supplement may add missing Core methods. Historical
        # snapshot-only protocols continue to schedule TimeSigma alone.
        protocol["baselines"] = supplemental_baselines(config.get("supplemental_baselines", []), defaults, baselines, public_snapshot)
    if config["schema"] == "zero_shot":
        from EasyTSAD.Controller import TSADController
        from timesigma.benchmark.runner import _working_directory

        with _working_directory(root.resolve()):
            controller = TSADController(log_level="error")
            controller.set_dataset(datasets=[config["dataset"]], dirname=str(dataset_root), dataset_type="UTS")
            membership_source = config.get('zero_shot_membership_source')
            if membership_source is not None:
                membership_source = Path(membership_source)
                if file_digest(membership_source) != config.get('zero_shot_membership_source_sha256'):
                    raise ValueError('approved native zero-shot membership source changed')
            split, = freeze_zero_shot_membership(controller, (config["dataset"],), root.resolve(),
                                                 expected_path=membership_source)
        if not split.training_curves or not split.held_out_curves:
            raise ValueError("zero-shot requires nonempty training and held-out members")
        protocol["zero_shot_split"] = split.as_dict()
        protocol["zero_shot_membership_sha256"] = file_digest(root / MEMBERSHIP_FILENAME)
        protocol["zero_shot_provenance"] = {
            "detector_training_exposure": "training_members_only",
            "checkpoint_selection_exposure": "training_members_only",
            "native_minmax_zscore_scaler_exposure": "per_curve_own_unlabeled_train_partition_including_held_out",
            "robust_scaler_exposure": "frozen_training_members_only",
            "ordering_source": "pinned EasyTSAD PathManager.get_dataset_curves / filesystem enumeration; exact order in zero_shot_split.enumerated_curves",
            "training_member_count": len(split.training_curves),
            "held_out_member_count": len(split.held_out_curves),
            "easytsad_revision": EASYTSAD_REVISION,
        }
    protocol["planning_units"] = planning_units(protocol)
    installed_source = json.loads(distribution("EasyTSAD").read_text("direct_url.json"))
    if installed_source.get("vcs_info", {}).get("commit_id") != EASYTSAD_REVISION:
        raise ValueError("installed EasyTSAD differs from the frozen Catalog revision")
    protocol["runtime_versions"] = {
        "python": sys.version.split()[0], "node": subprocess.check_output(["node", "--version"], text=True).strip(),
        "python_packages": {name: version(name) for name in ("EasyTSAD", "numpy", "torch", "scikit-learn")},
        "easytsad_source": installed_source,
        "pi_packages": {name: json.loads((REPOSITORY / "pi-host/node_modules/@earendil-works" / name / "package.json").read_text())["version"] for name in ("pi-ai", "pi-agent-core")},
    }
    protocol["capability_snapshot"] = lookup_capabilities(task_for(protocol, planning_units(protocol)[0]), kind="all", name="", one_shot=True)
    for path in source_files():
        target = root / "source" / path.relative_to(REPOSITORY)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json_mapping_if_same(root / "protocol.json", protocol)
    if recovery is not None:
        _recover_dry_run(recovery, root, previous, protocol)
    return protocol


def load_protocol(root: Path, *, verify_source: bool = True) -> dict[str, Any]:
    protocol = dict(read_json_mapping(root / "protocol.json"))
    if verify_source:
        frozen = root / "source"
        if sorted(p.relative_to(frozen) for p in frozen.rglob("*") if p.is_file()) != [p.relative_to(REPOSITORY) for p in source_files()]:
            raise ValueError("delivery source set changed after freeze")
        for path in source_files():
            if not filecmp.cmp(path, frozen / path.relative_to(REPOSITORY), shallow=False):
                raise ValueError(f"delivery source changed after freeze: {path.name}")
    if file_digest(root / "memory-snapshot.json") != protocol["memory_snapshot_sha256"]:
        raise ValueError("frozen Memory snapshot was changed")
    if protocol["schema"] == "zero_shot" and file_digest(root / MEMBERSHIP_FILENAME) != protocol["zero_shot_membership_sha256"]:
        raise ValueError("frozen zero-shot membership was changed")
    comparison = protocol.get("comparison")
    if comparison is not None and file_digest(Path(comparison["snapshot"])) != comparison["snapshot_sha256"]:
        raise ValueError("frozen public comparison snapshot was changed")
    return protocol


def frozen_memory(root: Path) -> EvidenceCollection:
    return _read_frozen_memory(read_json_mapping(root / "memory-snapshot.json"))


def _read_frozen_memory(snapshot: dict[str, Any]) -> EvidenceCollection:
    records = []
    for group in ("knowledge_files", "evidence_files"):
        for name, digest in snapshot[group].items():
            path = Path(name)
            if file_digest(path) != digest:
                raise ValueError(f"frozen Memory source changed: {name}")
            if group == "evidence_files":
                payload = read_json_mapping(path)
                cls = ComponentEvidence if payload["kind"] == "component" else ResearchStepEvidence
                records.append(cls.from_dict(payload))
    return EvidenceCollection(tuple(records))


def _recover_dry_run(source: Path, destination: Path, previous: dict[str, Any], current: dict[str, Any]) -> None:
    for key in ("dataset", "schema", "primary_metric", "curves", "seed", "provider", "planning", "default_pipelines", "baselines", "dataset_provenance", "memory_snapshot_sha256"):
        if previous[key] != current[key]:
            raise ValueError(f"engineering continuation would change frozen science: {key}")
    for key, value in previous["execution"].items():
        if current["execution"][key] != value:
            raise ValueError("engineering continuation cannot increase existing execution budgets")
    closed = read_json_mapping(source / "planning-closed.json")
    for name, digest in closed["plans"].items():
        if file_digest(source / name) != digest:
            raise ValueError("original plan was changed before recovery")
    shutil.copytree(source / "plans", destination / "plans")
    for curve in planning_units(current):
        path = destination / "plans" / curve / "final.json"
        plan = dict(read_json_mapping(path))
        plan["origin_plan_reference"] = str(source / "plans" / curve / "final.json")
        plan["origin_plan_sha256"] = file_digest(Path(plan["origin_plan_reference"]))
        plan["protocol_sha256"] = file_digest(destination / "protocol.json")
        path.write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if (source / "execution").exists():
        shutil.copytree(source / "execution", destination / "execution")
