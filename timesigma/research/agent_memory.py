from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import filecmp
import os
from pathlib import Path
import shutil
from typing import Any

from timesigma.algorithms import EASYTSAD_REVISION, EASYTSAD_SCHEMAS, METHOD_CATALOG
from timesigma.algorithms.artifacts import validate_score_lineage, validate_model_artifact_metadata
from timesigma.benchmark.manifest import dataset_provenance
from timesigma.benchmark import runtime
from timesigma.benchmark.capabilities.sources import DatasetSource, dataset_view
from timesigma.benchmark.zero_shot import ZeroShotSplit, load_zero_shot_membership
from timesigma.profiles.features import CURVE_PROFILE_DEFINITION

from .canonical import (
    ComponentEvidence,
    agent_memory_manifest_payload,
    component_evidence_id,
    zero_shot_membership_payload,
)
from .collection import (
    EvidenceCollection,
    evidence_directory,
    read_json_mapping,
    write_json_mapping_if_same,
    validate_component_tasks,
    plain_json,
)
from .pipeline import PipelineSpec, PlanningUnit
from .result_parser import parse_validated_component_results


@dataclass(frozen=True)
class AgentMemoryMaterializationReport:
    output_dir: str
    capability_run_dirs: int
    component_evidence: int
    research_step_evidence: int
    datasets: Mapping[str, int]
    schemas: Mapping[str, int]
    methods: Mapping[str, int]
    artifact_metadata: int
    artifact_files: int
    hardlinked_artifact_files: int
    copied_artifact_files: int
    replay_recipes: int
    upstream_final_state_unavailable: int
    zero_shot_datasets: int

    def as_dict(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "capability_run_dirs": self.capability_run_dirs,
            "component_evidence": self.component_evidence,
            "research_step_evidence": self.research_step_evidence,
            "datasets": dict(self.datasets),
            "schemas": dict(self.schemas),
            "methods": dict(self.methods),
            "artifact_metadata": self.artifact_metadata,
            "artifact_files": self.artifact_files,
            "hardlinked_artifact_files": self.hardlinked_artifact_files,
            "copied_artifact_files": self.copied_artifact_files,
            "replay_recipes": self.replay_recipes,
            "upstream_final_state_unavailable": self.upstream_final_state_unavailable,
            "zero_shot_datasets": self.zero_shot_datasets,
        }


@dataclass(frozen=True)
class AgentMemoryValidationReport:
    root: str
    component_evidence: int
    research_step_evidence: int
    aiops_evidence: int
    torch_artifacts: int
    torch_artifacts_loaded: int
    fitted_estimators: int
    fitted_estimators_loaded: int
    replay_recipes: int
    upstream_final_state_unavailable: int
    missing_artifacts: int
    absolute_path_violations: int
    manifest_ok: bool
    zero_shot_datasets: int

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "component_evidence": self.component_evidence,
            "research_step_evidence": self.research_step_evidence,
            "aiops_evidence": self.aiops_evidence,
            "torch_artifacts": self.torch_artifacts,
            "torch_artifacts_loaded": self.torch_artifacts_loaded,
            "fitted_estimators": self.fitted_estimators,
            "fitted_estimators_loaded": self.fitted_estimators_loaded,
            "replay_recipes": self.replay_recipes,
            "upstream_final_state_unavailable": self.upstream_final_state_unavailable,
            "missing_artifacts": self.missing_artifacts,
            "absolute_path_violations": self.absolute_path_violations,
            "manifest_ok": self.manifest_ok,
            "zero_shot_datasets": self.zero_shot_datasets,
        }


@dataclass(frozen=True)
class _ArtifactPlan:
    source_path: Path | None
    artifact_name: str | None
    metadata: Mapping[str, Any] | None


@dataclass(frozen=True)
class _PreparedEvidence:
    evidence: ComponentEvidence
    artifact: _ArtifactPlan


def materialize_agent_memory(
    *,
    output_dir: Path,
    dataset_roots: Sequence[Path],
    profile_root: Path,
    bootstrap_root: Path | None = None,
    methods: Sequence[str] = (),
    schemas: Sequence[str] = (),
    datasets: Sequence[str] = (),
    seed: int = 1,
) -> AgentMemoryMaterializationReport:
    if (output_dir / "manifest.json").exists():
        raise ValueError("bootstrap is for an initial store; incremental Method onboarding must reuse existing canonical tasks through qualification and admission")
    roots = tuple(path.resolve() for path in dataset_roots)
    if not roots:
        raise ValueError("at least one dataset root is required")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    dataset_sources = _bootstrap_dataset_sources(roots)
    selected_datasets = _select_known(tuple(dataset_sources), datasets, "dataset")
    selected_methods = _select_known(tuple(METHOD_CATALOG), methods, "method")
    selected_schemas = _select_known(EASYTSAD_SCHEMAS, schemas, "schema")
    selected_sources = {dataset: dataset_sources[dataset] for dataset in selected_datasets}

    output = output_dir.resolve()
    execution_root = (
        bootstrap_root.resolve()
        if bootstrap_root is not None
        else output.parent / f"{output.name}-bootstrap-runs"
    )
    if execution_root == output:
        raise ValueError("bootstrap execution root must be separate from output_dir")
    execution_root.mkdir(parents=True, exist_ok=True)

    jobs: list[Path] = []
    prepared: list[_PreparedEvidence] = []
    dataset_source_values: dict[str, str] = {}
    dataset_revisions: dict[str, str] = {}
    seeds: set[int] = set()

    with dataset_view(selected_sources) as easytsad_root:
        for method in selected_methods:
            spec = METHOD_CATALOG[method]
            for schema in selected_schemas:
                capability = spec.capability(schema)
                if not capability.supported:
                    continue
                for dataset in selected_datasets:
                    source = selected_sources[dataset]
                    job = execution_root / method / "default" / schema / dataset
                    _run_bootstrap_component_job(
                        method=method,
                        schema=schema,
                        dataset=dataset,
                        source=source,
                        easytsad_root=easytsad_root,
                        execution_root=execution_root,
                        job=job,
                        seed=seed,
                    )
                    parsed = parse_validated_component_results(
                        job,
                        easytsad_root / "UTS" / dataset,
                        source.provenance,
                        profile_dir=_profile_dir(
                            profile_root,
                            dataset=dataset,
                            source=_string(source.provenance, "source"),
                            revision=_string(source.provenance, "revision"),
                        ),
                    )
                    parsed_seed = parsed.evidence_reference.get("seed")
                    if isinstance(parsed_seed, bool) or not isinstance(parsed_seed, int):
                        raise ValueError(f"bootstrap seed is invalid: {job}")
                    seeds.add(parsed_seed)
                    dataset_source_values[dataset] = _string(source.provenance, "source")
                    dataset_revisions[dataset] = _string(source.provenance, "revision")
                    jobs.append(job / "run.json")
                    for unit in parsed.units:
                        prepared.append(_prepare_component_evidence(parsed, unit, job))

    if not prepared:
        raise ValueError("Agent Memory bootstrap found no supported Component results")
    if len(seeds) != 1:
        raise ValueError("Agent Memory bootstrap expects one fixed/default seed")
    membership = _zero_shot_membership(execution_root)
    _require_zero_shot_membership_consistency(jobs, membership)

    collection = EvidenceCollection(tuple(item.evidence for item in prepared))
    output.mkdir(parents=True, exist_ok=True)
    write_json_mapping_if_same(
        output / "manifest.json",
        agent_memory_manifest_payload(
            dataset_source=_common_dataset_source(dataset_source_values),
            dataset_sources=(
                None
                if _common_dataset_source(dataset_source_values) is not None
                else dict(sorted(dataset_source_values.items()))
            ),
            dataset_revisions=dict(sorted(dataset_revisions.items())),
            seed=next(iter(seeds)),
            zero_shot_membership="zero-shot-membership.json",
        ),
    )
    write_json_mapping_if_same(
        output / "zero-shot-membership.json",
        zero_shot_membership_payload(membership),
    )

    validate_component_tasks(output, collection.evidence)
    linked = 0
    copied = 0
    metadata_count = 0
    artifact_files = 0
    for item in prepared:
        directory = evidence_directory(output, item.evidence)
        if item.artifact.metadata is not None:
            metadata_count += 1
            write_json_mapping_if_same(directory / "artifact.json", item.artifact.metadata)
        if item.artifact.source_path is not None and item.artifact.artifact_name is not None:
            artifact_files += 1
            mode = _materialize_artifact_file(
                item.artifact.source_path,
                directory / item.artifact.artifact_name,
            )
            linked += int(mode == "linked")
            copied += int(mode == "copied")

    collection.write_to(output)
    return AgentMemoryMaterializationReport(
        output_dir=str(output),
        capability_run_dirs=len(jobs),
        component_evidence=sum(isinstance(item.evidence, ComponentEvidence) for item in prepared),
        research_step_evidence=0,
        datasets=Counter(str(evidence.task["dataset"]) for evidence in collection.evidence),
        schemas=Counter(str(evidence.task["schema"]) for evidence in collection.evidence),
        methods=Counter(str(evidence.pipeline["method"]) for evidence in collection.evidence),
        artifact_metadata=metadata_count,
        artifact_files=artifact_files,
        hardlinked_artifact_files=linked,
        copied_artifact_files=copied,
        replay_recipes=sum(
            item.evidence.artifact["artifact_type"] == "replay_recipe"
            for item in prepared
        ),
        upstream_final_state_unavailable=sum(
            item.evidence.artifact["selection_status"] == "unavailable"
            and item.evidence.artifact["score_source"] == "upstream_final_state"
            for item in prepared
        ),
        zero_shot_datasets=len(membership),
    )


def validate_agent_memory_store(
    root: Path,
    *,
    load_artifacts: bool = True,
) -> AgentMemoryValidationReport:
    directory = root.resolve()
    collection = EvidenceCollection.from_directory(directory)
    validate_component_tasks(directory, collection.evidence)
    manifest = read_json_mapping(directory / "manifest.json")
    membership = load_zero_shot_membership(directory / "zero-shot-membership.json")
    manifest_ok = (
        manifest.get("store") == "agent-memory"
        and manifest.get("easytsad_revision") == EASYTSAD_REVISION
        and manifest.get("zero_shot_membership") == "zero-shot-membership.json"
        and isinstance(manifest.get("dataset_revisions"), Mapping)
        and (
            isinstance(manifest.get("dataset_source"), str)
            or isinstance(manifest.get("dataset_sources"), Mapping)
        )
    )

    counts = Counter()
    missing_artifacts = 0
    absolute_path_violations = 0
    for evidence in collection.evidence:
        if str(evidence.task["dataset"]).casefold() == "aiops":
            counts["aiops"] += 1
        if _contains_absolute_path(evidence.as_dict()):
            absolute_path_violations += 1

        artifact = evidence.artifact
        artifact_type = str(artifact["artifact_type"])
        status = str(artifact["selection_status"])
        score_source = artifact["score_source"]
        validate_score_lineage(status, artifact_type, score_source)  # type: ignore[arg-type]
        artifact_path = artifact["artifact_path"]
        evidence_dir = evidence_directory(directory, evidence)
        pipeline = evidence.pipeline if isinstance(evidence, ComponentEvidence) else evidence.candidate_pipeline
        if pipeline.get("operators") or pipeline.get("ensemble_ref"):
            from .pipeline_execution import validate_pipeline_bundle

            validate_pipeline_bundle(evidence_dir, pipeline)
        validate_evidence_artifact(directory, evidence, manifest=manifest)
        artifact_metadata = evidence_dir / "artifact.json"
        if artifact_metadata.exists() and _contains_absolute_path(
            read_json_mapping(artifact_metadata)
        ):
            absolute_path_violations += 1

        if artifact_type == "torch_state_dict":
            if status in ("available", "not_applicable"):
                counts["torch"] += 1
                expected_name = "final.pt" if status == "not_applicable" else "best.pt"
                if artifact_path != expected_name:
                    missing_artifacts += 1
                    continue
                model_path = evidence_dir / expected_name
                if not model_path.is_file():
                    missing_artifacts += 1
                    continue
                if load_artifacts:
                    _load_torch_artifact(model_path)
                    counts["torch_loaded"] += 1
            elif status == "unavailable" and score_source == "upstream_final_state":
                counts["upstream_final_state_unavailable"] += 1
                if artifact_path is not None:
                    missing_artifacts += 1
        elif artifact_type == "fitted_estimator":
            counts["fitted"] += 1
            if artifact_path != "estimator.pkl":
                missing_artifacts += 1
                continue
            model_path = evidence_dir / "estimator.pkl"
            if not model_path.is_file():
                missing_artifacts += 1
                continue
            if load_artifacts:
                _load_fitted_estimator(model_path)
                counts["fitted_loaded"] += 1
        elif artifact_type == "replay_recipe":
            counts["replay"] += 1
            if artifact_path is not None:
                missing_artifacts += 1
            if (evidence_dir / "best.pt").exists() or (evidence_dir / "estimator.pkl").exists():
                missing_artifacts += 1

    component_count = sum(
        1 for item in collection.evidence if isinstance(item, ComponentEvidence)
    )
    return AgentMemoryValidationReport(
        root=str(directory),
        component_evidence=component_count,
        research_step_evidence=len(collection.evidence) - component_count,
        aiops_evidence=counts["aiops"],
        torch_artifacts=counts["torch"],
        torch_artifacts_loaded=counts["torch_loaded"],
        fitted_estimators=counts["fitted"],
        fitted_estimators_loaded=counts["fitted_loaded"],
        replay_recipes=counts["replay"],
        upstream_final_state_unavailable=counts["upstream_final_state_unavailable"],
        missing_artifacts=missing_artifacts,
        absolute_path_violations=absolute_path_violations,
        manifest_ok=manifest_ok,
        zero_shot_datasets=len(membership),
    )


def validate_evidence_artifact(
    root: Path,
    evidence: Any,
    *,
    manifest: Mapping[str, Any] | None = None,
    artifact_directory: Path | None = None,
) -> Mapping[str, Any] | None:
    """Bind compact historical metadata to its existing Evidence/store identity.

    Compact records intentionally obtain Method/configuration and old seed/revision
    from their canonical owners; this does not recover original raw-score provenance.
    """
    from timesigma.algorithms import get_method
    from timesigma.algorithms.configuration import resolve_configuration
    from .pipeline import NormalizedPipeline
    from .planning import pipeline_detectors

    manifest = read_json_mapping(root / "manifest.json") if manifest is None else manifest
    task = plain_json(evidence.task)
    dataset, schema = task["dataset"], task["schema"]
    if manifest.get("easytsad_revision") != EASYTSAD_REVISION or dataset not in manifest["dataset_revisions"]:
        raise ValueError("Evidence source differs from canonical store provenance")
    if type(manifest.get("seed")) is not int or manifest["seed"] < 0:
        raise ValueError("canonical store seed is invalid")
    pipeline = plain_json(evidence.pipeline if isinstance(evidence, ComponentEvidence) else evidence.candidate_pipeline)
    normalized = NormalizedPipeline.from_dict(pipeline)
    for detector in pipeline_detectors(normalized):
        spec = get_method(detector.method)
        defaults = resolve_configuration(spec.module, (dataset,))[dataset]["Model_Params"]
        spec.validate_parameters(detector.as_dict()["resolved_configuration"]["Model_Params"], defaults)
    directory = artifact_directory or evidence_directory(root, evidence)
    if pipeline.get("operators") or pipeline.get("ensemble_ref"):
        from .pipeline_execution import validate_pipeline_bundle

        binding = validate_pipeline_bundle(directory, pipeline)
        bound_task = binding["task"]
        if any(bound_task[key] != task[key] for key in ("dataset", "schema", "unit")) or binding["seed"] != manifest["seed"]:
            raise ValueError("Pipeline Artifact task or seed differs from its Evidence")
        return binding
    spec = get_method(pipeline["method"])
    artifact = plain_json(evidence.artifact)
    if artifact["artifact_type"] != spec.artifact_type:
        raise ValueError("Evidence Artifact type differs from its Method")
    validate_score_lineage(artifact["selection_status"], spec.artifact_type, artifact["score_source"])
    metadata_path = directory / "artifact.json"
    # A native training-free Component has no additional compact model metadata.
    if not metadata_path.is_file() and spec.artifact_type == "replay_recipe":
        if artifact["artifact_path"] is not None:
            raise ValueError("replay recipe cannot require a model binary")
        return None
    if not metadata_path.is_file():
        raise ValueError("required Component/ResearchStep Artifact metadata is missing")
    metadata = dict(read_json_mapping(metadata_path))
    identity = {
        "method": pipeline["method"], "schema": schema, "dataset": dataset,
        "curve": task["unit"]["curve"], "configuration": pipeline["resolved_configuration"],
        "seed": manifest["seed"], "dataset_revision": manifest["dataset_revisions"][dataset],
        "easytsad_revision": manifest["easytsad_revision"],
    }
    for key, value in identity.items():
        if key in metadata and (metadata[key] != value or (key == "seed" and type(metadata[key]) is not int)):
            raise ValueError(f"Artifact {key} differs from canonical Evidence/store identity")
    # Required new-method seed must remain present in the physical metadata.
    if spec.method_name in ("LSTMADalpha", "LSTMADbeta", "FITS", "DCdetector", "FCVAE") and "seed" not in metadata:
        raise ValueError("selected Artifact requires its frozen seed")
    metadata = {**identity, **metadata}
    if any(metadata.get(key) != value for key, value in artifact.items()):
        raise ValueError("Artifact selection facts differ from canonical Evidence")
    training = task["unit"]["training_curves"]
    testing = task["unit"]["held_out_curves"] if schema == "zero_shot" else training
    if metadata.get("training_curves") != training or metadata.get("test_curves") != testing:
        raise ValueError("Artifact execution order differs from canonical task order")
    if schema == "zero_shot":
        splits = load_zero_shot_membership(root / manifest["zero_shot_membership"])
        split = next((item for item in splits if item.dataset == dataset), None)
        if split is None or metadata.get("zero_shot_split") != split.as_dict():
            raise ValueError("Artifact differs from canonical store-level zero-shot membership")
    try:
        validate_model_artifact_metadata(spec, metadata, schema=schema, configuration=pipeline["resolved_configuration"])
    except ValueError as exc:
        raise ValueError(f"Artifact checkpoint contract differs: {exc}") from exc
    model = artifact["artifact_path"]
    if model is not None and not (directory / model).is_file():
        raise ValueError("existing Component Artifact is missing; cannot skip execution")
    if spec.method_name == "IsolationForest":
        import joblib
        from timesigma.algorithms.isolation_forest.binding import validate_estimator

        validate_estimator(joblib.load(directory / model), metadata)
    return metadata


def _run_bootstrap_component_job(
    *,
    method: str,
    schema: str,
    dataset: str,
    source: DatasetSource,
    easytsad_root: Path,
    execution_root: Path,
    job: Path,
    seed: int,
) -> None:
    run_path = job / "run.json"
    if run_path.is_file():
        manifest = read_json_mapping(run_path)
        if manifest.get("status") == "complete":
            _publish_or_check_bootstrap_membership(schema, dataset, job, execution_root)
            return
        raise ValueError(f"bootstrap job is incomplete: {job}")

    expected_split = _bootstrap_membership_path(execution_root, dataset)
    results_dir, _ = runtime.run_method(
        method_name=method,
        schema=schema,
        datasets=(dataset,),
        dataset_root=easytsad_root,
        dataset_source_root=source.root,
        output_dir=job,
        configuration_name="default",
        zero_shot_membership=(
            expected_split if schema == "zero_shot" and expected_split.is_file() else None
        ),
        # Only an initial store may generate membership. Preserve the native
        # enumeration once, then record it for the parser; incremental runs use tasks.
        **({"ordered_members": {dataset: tuple(os.listdir(easytsad_root / "UTS" / dataset))}}
           if schema == "all_in_one" else {}),
        seed=seed,
        model_artifacts=True,
    )
    if results_dir.resolve() != (job / "Results").resolve():
        raise RuntimeError("EasyTSAD returned an unexpected Results directory")
    _publish_or_check_bootstrap_membership(schema, dataset, job, execution_root)


def _prepare_component_evidence(
    parsed: Any,
    unit: Any,
    job: Path,
) -> _PreparedEvidence:
    artifact_reference = unit.artifact_reference
    source_metadata = read_json_mapping(job / str(artifact_reference["metadata"]))
    artifact_type = str(artifact_reference["artifact_type"])
    selection_status = str(artifact_reference["selection_status"])
    score_source = artifact_reference["score_source"]
    validate_score_lineage(selection_status, artifact_type, score_source)  # type: ignore[arg-type]
    if selection_status == "available" and artifact_type == "torch_state_dict" and score_source != "selected_artifact":
        raise ValueError("new Component metrics must come from the selected Artifact; preserve upstream-final comparisons as separate execution facts")

    source_path: Path | None = None
    artifact_name: str | None = None
    if artifact_type == "torch_state_dict" and selection_status == "available":
        source_path = job / str(artifact_reference["state"])
        artifact_name = "best.pt"
    elif artifact_type == "torch_state_dict" and selection_status == "not_applicable":
        source_path = job / str(artifact_reference["state"])
        artifact_name = "final.pt"
    elif artifact_type == "fitted_estimator":
        source_path = job / str(artifact_reference["state"])
        artifact_name = "estimator.pkl"
    elif artifact_type == "torch_state_dict" and selection_status == "unavailable":
        if score_source != "upstream_final_state":
            raise ValueError("unavailable torch Evidence must use upstream_final_state")
    elif artifact_type != "replay_recipe":
        raise ValueError(
            f"unsupported artifact bootstrap state: {parsed.method}/{parsed.schema}/{parsed.dataset}"
        )

    if source_path is not None:
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Artifact file is missing: {source_path}")

    task = _task_from_parsed_unit(parsed, unit)
    pipeline = PipelineSpec(
        method=parsed.method,
        resolved_configuration=parsed.resolved_configuration,
    ).normalized().canonical_payload()
    evidence = ComponentEvidence(
        evidence_id=component_evidence_id(method=parsed.method, task=task),
        task=task,
        pipeline=pipeline,
        metrics=unit.metrics,
        artifact={
            "artifact_type": artifact_type,
            "selection_status": selection_status,
            "score_source": score_source,
            "artifact_path": artifact_name,
        },
    )
    metadata = (
        None
        if artifact_type == "replay_recipe"
        else _artifact_metadata(source_metadata, artifact_name)
    )
    return _PreparedEvidence(
        evidence=evidence,
        artifact=_ArtifactPlan(
            source_path=source_path,
            artifact_name=artifact_name,
            metadata=metadata,
        ),
    )


def _task_from_parsed_unit(parsed: Any, unit: Any) -> dict[str, Any]:
    planning_unit = PlanningUnit.for_schema(
        parsed.schema,
        curve=unit.curve,
        training_curves=unit.training_curves,
        held_out_curves=unit.held_out_curves,
    )
    return {
        "dataset": parsed.dataset,
        "schema": parsed.schema,
        "unit": planning_unit.as_dict(),
    }


def _artifact_metadata(
    source: Mapping[str, Any],
    artifact_name: str | None,
) -> dict[str, Any]:
    payload = {
        "artifact_path": artifact_name,
        "artifact_type": source.get("artifact_type"),
        "best_epoch": source.get("best_epoch"),
        "sample_counts": deepcopy(source.get("sample_counts")),
        "score_source": source.get("score_source"),
        "selection_objective": source.get("selection_objective"),
        "selection_policy": source.get("selection_policy"),
        "selection_reason": source.get("selection_reason"),
        "selection_status": source.get("selection_status"),
        "selection_values": deepcopy(source.get("selection_values")),
        "training_curves": deepcopy(source.get("training_curves")),
        "test_curves": deepcopy(source.get("test_curves")),
    }
    if source.get("zero_shot_split") is not None:
        payload["zero_shot_split"] = deepcopy(source.get("zero_shot_split"))
    if source.get("source_identity") == "timesigma-selected":
        payload.update({key: deepcopy(source[key]) for key in ("source_identity", "seed", "score_support")})
    if source.get("method") == "IsolationForest" or source.get("source_identity") == "sklearn.ensemble.IsolationForest@1.9.0":
        from timesigma.algorithms.isolation_forest.binding import BINDING_FIELDS

        payload.update({key: deepcopy(source[key]) for key in BINDING_FIELDS})
    return payload


def _materialize_artifact_file(source: Path, target: Path) -> str:
    if target.exists():
        if filecmp.cmp(source, target, shallow=False):
            return "existing"
        raise ValueError(f"refusing to overwrite different artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "linked"
    except OSError:
        shutil.copy2(source, target)
        return "copied"


def _zero_shot_membership(execution_root: Path) -> tuple[ZeroShotSplit, ...]:
    splits: list[ZeroShotSplit] = []
    for path in sorted((execution_root / "zero-shot-membership").glob("*.json")):
        loaded = load_zero_shot_membership(path)
        if len(loaded) != 1:
            raise ValueError(f"shared zero-shot membership must contain one dataset: {path}")
        splits.append(loaded[0])
    if len({split.dataset for split in splits}) != len(splits):
        raise ValueError("duplicate dataset in shared zero-shot membership")
    return tuple(sorted(splits, key=lambda split: split.dataset))


def _require_zero_shot_membership_consistency(
    jobs: Sequence[Path],
    membership: Sequence[ZeroShotSplit],
) -> None:
    by_dataset = {split.dataset: split for split in membership}
    for run_path in jobs:
        manifest = read_json_mapping(run_path)
        if manifest.get("schema") != "zero_shot":
            continue
        dataset = _single_dataset(manifest)
        expected = by_dataset.get(dataset)
        if expected is None:
            raise ValueError(f"missing shared zero-shot membership for {dataset}")
        path = run_path.parent / str(manifest["zero_shot_membership_path"])
        loaded = load_zero_shot_membership(path)
        if loaded != (expected,):
            raise ValueError(f"per-run zero-shot membership differs: {run_path.parent}")


def _publish_or_check_bootstrap_membership(
    schema: str,
    dataset: str,
    job: Path,
    execution_root: Path,
) -> None:
    if schema != "zero_shot":
        return
    source = job / "Results" / "zero_shot_membership.json"
    target = _bootstrap_membership_path(execution_root, dataset)
    if target.is_file():
        if load_zero_shot_membership(source) != load_zero_shot_membership(target):
            raise ValueError(f"zero-shot membership differs for {dataset}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _bootstrap_membership_path(execution_root: Path, dataset: str) -> Path:
    return execution_root / "zero-shot-membership" / f"{dataset}.json"


def _bootstrap_dataset_sources(roots: Sequence[Path]) -> dict[str, DatasetSource]:
    sources: dict[str, DatasetSource] = {}
    for root in roots:
        provenance = dataset_provenance(root)
        if (
            provenance.get("source") is None
            or provenance.get("revision") is None
            or provenance.get("clean") is not True
        ):
            raise ValueError(f"dataset source must be a clean git checkout: {root}")
        for dataset, relative_path in _dataset_candidates(root):
            if dataset.casefold() == "aiops":
                continue
            source = DatasetSource(
                root=root,
                provenance=provenance,
                relative_path=relative_path,
            )
            existing = sources.get(dataset)
            if existing is not None and (
                existing.root != source.root
                or existing.relative_path != source.relative_path
                or dict(existing.provenance) != dict(source.provenance)
            ):
                raise ValueError(f"dataset source is ambiguous for {dataset}")
            sources[dataset] = source
    return dict(sorted(sources.items()))


def _dataset_candidates(root: Path) -> tuple[tuple[str, str], ...]:
    uts = root / "UTS"
    if uts.is_dir():
        return tuple(
            sorted(
                (path.name, f"UTS/{path.name}")
                for path in uts.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            )
        )
    if _looks_like_uts_dataset(root):
        return ((root.name, "."),)
    return tuple(
        sorted(
            (path.name, path.name)
            for path in root.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
            and _looks_like_uts_dataset(path)
        )
    )


def _looks_like_uts_dataset(path: Path) -> bool:
    return any(
        curve.is_dir() and (curve / "train.npy").is_file()
        for curve in path.iterdir()
    )


def _profile_dir(
    profile_root: Path,
    *,
    dataset: str,
    source: str,
    revision: str,
) -> Path:
    candidates = (
        profile_root / dataset / revision,
        profile_root,
    )
    matches: list[Path] = []
    for candidate in candidates:
        manifest_path = candidate / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = read_json_mapping(manifest_path)
        if (
            manifest.get("dataset") == dataset
            and manifest.get("source") == source
            and manifest.get("revision") == revision
            and manifest.get("profile_definition") == CURVE_PROFILE_DEFINITION
        ):
            matches.append(candidate.resolve())
    unique = tuple(dict.fromkeys(matches))
    if len(unique) != 1:
        raise ValueError(f"profile source is missing or ambiguous for {dataset}")
    return unique[0]


def _common_dataset_source(sources: Mapping[str, str]) -> str | None:
    values = set(sources.values())
    return next(iter(values)) if len(values) == 1 else None


def _single_dataset(manifest: Mapping[str, Any]) -> str:
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1 or not isinstance(datasets[0], str):
        raise ValueError("bootstrap run must contain exactly one dataset")
    return datasets[0]


def _select_known(
    available: Sequence[str],
    requested: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    values = tuple(requested)
    if not values:
        return tuple(available)
    if len(set(values)) != len(values) or set(values) - set(available):
        raise ValueError(f"Agent Memory bootstrap selection contains an invalid {label}")
    return tuple(item for item in available if item in set(values))


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        return Path(value).is_absolute()
    return False


def _load_torch_artifact(path: Path) -> None:
    import torch

    try:
        torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        torch.load(path, map_location="cpu")


def _load_fitted_estimator(path: Path) -> None:
    import joblib

    joblib.load(path)


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


__all__ = [
    "AgentMemoryMaterializationReport",
    "AgentMemoryValidationReport",
    "materialize_agent_memory",
    "validate_agent_memory_store",
]
