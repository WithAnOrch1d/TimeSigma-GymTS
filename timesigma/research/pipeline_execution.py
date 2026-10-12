from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from scipy.stats import rankdata
import toml

from timesigma.algorithms import get_method
from timesigma.algorithms.configuration import freeze_configuration, resolve_configuration
from timesigma.benchmark import runtime
from timesigma.benchmark.runner import SUMMARY_ITEMS, _working_directory, official_evaluations
from timesigma.benchmark.zero_shot import MEMBERSHIP_FILENAME

from .collection import read_json_mapping, write_json_mapping_if_same
from .pipeline import NormalizedPipeline, TaskContext
from .planning import pipeline_detectors, validate_planned_pipeline
from .pipeline_transforms import fuse_scores_mean
from .result_parser import _ParsedComponentResult, _official_metrics, parse_validated_component_results


def run_planned_pipeline(
    plan: Any,
    *,
    dataset_root: Path,
    output_dir: Path,
    dataset_source_root: Path | None,
    zero_shot_membership: Path | None,
    evaluate: bool = True,
    model_artifacts: bool = True,
) -> tuple[Path, Path]:
    from EasyTSAD.Controller import TSADController

    pipeline = plan.pipeline.normalized()
    validate_planned_pipeline(pipeline, plan.task)
    if len(pipeline_detectors(pipeline)) > plan.budget["max_runs"]:
        raise ValueError("Pipeline exceeds the frozen detector execution budget")
    root = output_dir.resolve()
    definition = pipeline.canonical_payload()
    freeze_configuration(
        root / "configuration.toml",
        {plan.task.dataset: definition["resolved_configuration"]},
        pipeline={"definition": json.dumps(definition, sort_keys=True)},
    )
    branches = []
    runtime_python = None
    for index, detector in enumerate(pipeline_detectors(pipeline)):
        branch = root / "branches" / str(index)
        inputs = pipeline_input_configuration(pipeline, plan.task)
        configuration = freeze_configuration(
            root / f"branch-{index}.toml",
            {plan.task.dataset: detector.as_dict()["resolved_configuration"]},
            pipeline=inputs,
        )
        _, runtime_python = runtime.run_method(
            method_name=detector.method,
            schema=plan.task.schema,
            datasets=(plan.task.dataset,),
            dataset_root=dataset_root,
            dataset_source_root=dataset_source_root,
            output_dir=branch,
            configuration_name=plan.experiment_id,
            configuration_path=configuration,
            zero_shot_membership=zero_shot_membership,
            curve_names=(plan.task.unit.curve,) if plan.task.schema == "naive" else None,
            **({"ordered_members": {plan.task.dataset: plan.task.unit.training_curves}} if plan.task.schema == "all_in_one" else {}),
            seed=plan.seed,            model_artifacts=model_artifacts,
            evaluate=False, label_free=True,
        )
        branches.append({
            "directory": branch.relative_to(root).as_posix(),
            "method": detector.method,
            "pipeline_inputs": inputs,
        })

    curves = (
        plan.task.unit.held_out_curves
        if plan.task.schema == "zero_shot" else plan.task.unit.training_curves
    )
    destination = root / "Results" / "Scores" / "TimeSigma" / plan.task.schema / plan.task.dataset
    destination.mkdir(parents=True, exist_ok=True)
    support = {}
    for curve in curves:
        scores = [
            np.load(
                root / item["directory"] / "Results" / "Scores" / item["method"]
                / plan.task.schema / plan.task.dataset / f"{curve}.npy",
                allow_pickle=False,
            )
            for item in branches
        ]
        test_length = len(np.load(
            dataset_root / "UTS" / plan.task.dataset / curve / "test.npy",
            mmap_mode="r", allow_pickle=False,
        ))
        for detector, score in zip(pipeline_detectors(pipeline), scores):
            if (
                score.ndim != 1 or not score.size
                or not np.isfinite(score).all() or score.size > test_length
            ):
                raise ValueError("Pipeline branch produced invalid scores")
            if pipeline.ensemble_ref is not None:
                expected = get_method(detector.method).sample_spec.count(
                    test_length,
                    SimpleNamespace(**detector.as_dict()["resolved_configuration"]["Model_Params"]),
                )
                if score.size != expected:
                    raise ValueError("branch score violates its native suffix alignment contract")
        if len(scores) == 2:
            if scores[0].shape != scores[1].shape:
                raise ValueError("mean fusion requires identical native score support")
            scores = [(rankdata(score, method="average") - 0.5) / len(score) for score in scores]
        fused = fuse_scores_mean(scores)
        np.save(destination / f"{curve}.npy", fused, allow_pickle=False)
        support[curve] = {"start": test_length - len(fused), "stop": test_length}

    if zero_shot_membership is not None:
        shutil.copy2(
            root / branches[0]["directory"] / "Results" / MEMBERSHIP_FILENAME,
            root / "Results" / MEMBERSHIP_FILENAME,
        )
    if evaluate:
        # All branches and the final fused score exist before any label is loaded.
        for branch in branches:
            evaluate_pipeline_scores(root / branch["directory"], plan.task, dataset_root, method=branch["method"])
        evaluate_pipeline_scores(root, plan.task, dataset_root)

    files = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*")
        if path.is_file() and path.name != "TSADEval.log"
    )
    digests = {
        name: file_digest(root / name) for name in files
        if Path(name).suffix in {".npy", ".pt", ".pkl"}
        or Path(name).name == "preprocessing-state.json"
    }
    write_json_mapping_if_same(root / "pipeline-artifact.json", {
        "pipeline": definition, "task": plan.task.as_dict(), "seed": plan.seed,
        "branches": branches, "score_support": support,
        "files": files, "digests": digests,
    })
    write_json_mapping_if_same(root / "run.json", {
        "status": "complete", "pipeline": definition,
        "task": plan.task.as_dict(), "seed": plan.seed,
        "experiment_id": plan.experiment_id,
    })
    assert runtime_python is not None
    return root / "Results", runtime_python


def evaluate_pipeline_scores(root: Path, task: TaskContext, dataset_root: Path, *, method: str = "TimeSigma") -> None:
    from EasyTSAD.Controller import TSADController

    with _working_directory(root):
        controller = TSADController(log_level="error")
        scope = (
            {"specify_curves": True, "curve_names": [task.unit.curve]}
            if task.schema == "naive" else {}
        )
        controller.set_dataset(
            datasets=task.dataset if scope else [task.dataset],
            dirname=str(dataset_root.resolve()), dataset_type="UTS", **scope,
        )
        controller.set_evals(official_evaluations())
        controller.do_evals(method=method, training_schema=task.schema)
        controller.summary.to_csv(
            datasets=[task.dataset], methods=[method],
            training_schema=task.schema,
            eval_items=[list(item) for item in SUMMARY_ITEMS],
        )

def validate_pipeline_bundle(root: Path, pipeline: Mapping[str, Any]) -> Mapping[str, Any]:
    binding = read_json_mapping(root / "pipeline-artifact.json")
    if binding["pipeline"] != NormalizedPipeline.from_dict(pipeline).canonical_payload():
        raise ValueError("Pipeline Artifact belongs to a different complete Pipeline")
    configuration = toml.load(root / "configuration.toml")
    if json.loads(configuration["Pipeline"]["definition"]) != binding["pipeline"]:
        raise ValueError("frozen Pipeline configuration does not match its Artifact")
    dataset = binding["task"]["dataset"]
    native = resolve_configuration(
        get_method(binding["pipeline"]["method"]).module, (dataset,), configuration,
    )[dataset]
    if native != binding["pipeline"]["resolved_configuration"]:
        raise ValueError("native configuration projection does not match the complete Pipeline")
    for name in binding["files"]:
        if not bundle_file(root, name).is_file():
            raise FileNotFoundError(f"Pipeline Artifact file is missing: {name}")
    for name, digest in binding["digests"].items():
        if file_digest(bundle_file(root, name)) != digest:
            raise ValueError(f"Pipeline Artifact integrity mismatch: {name}")
    return binding


def pipeline_input_configuration(
    pipeline: NormalizedPipeline, task: TaskContext,
) -> dict[str, Any]:
    # JSON preserves empty component parameter objects across TOML workers.
    return {
        "operators_json": json.dumps(pipeline.canonical_payload()["operators"], sort_keys=True),
        "schema": task.schema,
        "dataset": task.dataset,
        "training_curves": list(task.unit.training_curves),
        "held_out_curves": list(task.unit.held_out_curves),
    }


def parse_planned_result(
    plan: Any,
    root: Path,
    dataset_dir: Path,
    provenance: Mapping[str, Any],
    profile_dir: Path,
) -> _ParsedComponentResult:
    binding = validate_pipeline_bundle(root, plan.pipeline.normalized().canonical_payload())
    if binding["task"] != plan.task.as_dict() or binding["seed"] != plan.seed:
        raise ValueError("Pipeline Artifact execution context mismatch")
    results = []
    detectors = pipeline_detectors(plan.pipeline.normalized())
    if len(binding["branches"]) != len(detectors):
        raise ValueError("Pipeline Artifact branch count mismatch")
    for detector, branch in zip(detectors, binding["branches"]):
        expected_inputs = pipeline_input_configuration(plan.pipeline.normalized(), plan.task)
        if branch["pipeline_inputs"] != expected_inputs:
            raise ValueError("branch inputs do not match the complete frozen Pipeline")
        directory = bundle_file(root, branch["directory"])
        parsed = parse_validated_component_results(
            directory, dataset_dir, provenance, profile_dir=profile_dir,
        )
        if (
            parsed.method != detector.method
            or parsed.configuration_name != plan.experiment_id
            or parsed.resolved_configuration != detector.as_dict()["resolved_configuration"]
            or parsed.evidence_reference["seed"] != plan.seed
        ):
            raise ValueError("Pipeline branch does not match its frozen configuration")
        if (
            toml.load(directory / "configuration.toml").get("Pipeline") != expected_inputs
            or read_json_mapping(directory / "run.json").get("pipeline_inputs") != expected_inputs
        ):
            raise ValueError("Pipeline input configuration mismatch")
        expected = plan.task.unit
        if len(parsed.units) != 1:
            raise ValueError("Pipeline branch must produce exactly one PlanningUnit")
        result = parsed.units[0]
        if (result.curve, result.training_curves, result.held_out_curves) != (expected.curve, expected.training_curves, expected.held_out_curves):
            raise ValueError("Pipeline branch PlanningUnit mismatch")
        results.append(result)
    evaluation = read_json_mapping(
        root / "Results" / "Evals" / "TimeSigma" / plan.task.schema / plan.task.dataset
        / ("all.json" if plan.task.schema == "naive" else "avg.json")
    )
    if plan.task.schema == "naive":
        evaluation = evaluation[plan.task.unit.curve]
    artifact = {
        "artifact_type": "replay_recipe",
        "selection_status": "not_applicable",
        "score_source": "recipe_execution",
        "state": None,
        "metadata": None,
        "pipeline_bundle": "pipeline-artifact.json",
    }
    return replace(
        results[0], metrics=_official_metrics(plan.task.dataset, evaluation),
        artifact_reference=artifact,
    )


def bundle_file(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Pipeline Artifact references must be relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Pipeline Artifact reference escapes its root")
    return resolved


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
