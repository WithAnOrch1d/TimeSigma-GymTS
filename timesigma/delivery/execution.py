from __future__ import annotations

import csv
from contextlib import nullcontext
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from timesigma.benchmark.leaderboard import OFFICIAL_METRIC_IDS
from timesigma.benchmark.manifest import dataset_provenance
from timesigma.benchmark.zero_shot import MEMBERSHIP_FILENAME
from timesigma.benchmark.runner import SUMMARY_ITEMS, _working_directory, official_evaluations
from timesigma.research.collection import read_json_mapping, write_json_mapping_if_same
from timesigma.research.pipeline import NormalizedPipeline
from timesigma.research.pipeline_execution import file_digest, run_planned_pipeline, validate_pipeline_bundle
from timesigma.research.planning import pipeline_detectors
from timesigma.research.result_parser import _official_metrics

from .protocol import REPOSITORY, evaluation_curves, load_protocol, planning_units, task_for
from .resources import bounded_process, cuda_resource_usage


def uses_cuda(pipeline: NormalizedPipeline) -> bool:
    """Preserve the qualified neural runtime when the Agent selects a Core12 method."""
    gpu_methods = {'AnomalyTransformer', 'LSTMADalpha', 'LSTMADbeta', 'FITS', 'FCVAE', 'DCdetector'}
    return any(detector.method in gpu_methods for detector in pipeline_detectors(pipeline))


def verify_closed_plans(root: Path) -> dict[str, Any]:
    protocol = load_protocol(root)
    closed = read_json_mapping(root / "planning-closed.json")
    expected = {f"plans/{curve}/final.json" for curve in planning_units(protocol)}
    if set(closed["plans"]) != expected or closed["protocol_sha256"] != file_digest(root / "protocol.json"):
        raise ValueError("all frozen target plans must close before execution")
    for name, digest in closed["plans"].items():
        if file_digest(root / name) != digest:
            raise ValueError("a target plan changed after planning closed")
    provenance = dataset_provenance(Path(protocol["dataset_source_root"]))
    if provenance != protocol["dataset_provenance"]:
        raise ValueError("frozen dataset source changed")
    return protocol


def variants(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return [*protocol["baselines"], {"name": protocol["variant"], "model_artifacts": True}]


def resolved_job(root: Path, protocol: dict[str, Any], name: str, curve: str):
    row = next((r for r in variants(protocol) if r["name"] == name), None)
    if row is None:
        raise ValueError("variant is outside the fixed baseline/Agent set")
    pipeline = (
        read_json_mapping(root / "plans" / curve / "final.json")["pipeline"]
        if name == protocol["variant"] else protocol["default_pipelines"][row["method"]]
    )
    return row, NormalizedPipeline.from_dict(pipeline)


def execute_worker(root: Path, name: str, curve: str, attempt: int) -> None:
    protocol = verify_closed_plans(root)
    if (root / "scores-frozen.json").exists():
        raise ValueError("scores are frozen; no target execution may be repeated")
    row, pipeline = resolved_job(root, protocol, name, curve)
    if not 1 <= attempt <= protocol["execution"]["max_attempts_per_job"]:
        raise ValueError("engineering retry budget exhausted")
    job = root / "execution" / name / curve / f"attempt-{attempt}"
    plan = SimpleNamespace(
        pipeline=pipeline, task=task_for(protocol, curve), seed=protocol["seed"],
        experiment_id=name, budget={"max_runs": len(pipeline_detectors(pipeline))},
    )
    with cuda_resource_usage(job / "resource-usage.json", protocol["execution"]["gpu_memory_mb"]) if uses_cuda(pipeline) else nullcontext():
        run_planned_pipeline(
            plan, dataset_root=Path(protocol["dataset_root"]), output_dir=job,
            dataset_source_root=Path(protocol["dataset_source_root"]),
            zero_shot_membership=root / MEMBERSHIP_FILENAME if protocol["schema"] == "zero_shot" else None,
            evaluate=False, model_artifacts=row["model_artifacts"],
        )


def execute_all(root: Path, *, retry_failed: bool = False, variant_names: list[str] | None = None) -> dict[str, Any]:
    protocol = verify_closed_plans(root)
    if (root / "scores-frozen.json").exists():
        raise ValueError("scores are already frozen")
    failures = []
    rows = variants(protocol)
    if variant_names is not None:
        if not variant_names or len(variant_names) != len(set(variant_names)) or set(variant_names) - {row["name"] for row in rows}:
            raise ValueError("execution subset must contain distinct frozen variants")
        rows = [row for row in rows if row["name"] in variant_names]
    for row in rows:
        for curve in planning_units(protocol):
            directory = root / "execution" / row["name"] / curve
            outcomes = sorted(directory.glob("outcome-*.json"))
            if outcomes:
                previous = read_json_mapping(outcomes[-1])
                if previous["status"] == "complete":
                    continue
                if not retry_failed:
                    failures.append({"variant": row["name"], "curve": curve, "reason": previous["reason"]})
                    continue
            attempt = len(outcomes) + 1
            if attempt > protocol["execution"]["max_attempts_per_job"]:
                raise ValueError("frozen engineering retry budget exhausted")
            _, pipeline = resolved_job(root, protocol, row["name"], curve)
            branches = len(pipeline_detectors(pipeline))
            directory.mkdir(parents=True, exist_ok=True)
            command = [str(REPOSITORY / ".venv/bin/python"), "-m", "timesigma.delivery.cli", "worker", str(root.resolve()), row["name"], curve, str(attempt)]
            log = directory / f"attempt-{attempt}.log"
            record = {"variant": row["name"], "curve": curve, "attempt": attempt, "reserved_detector_executions": branches}
            try:
                outcome = bounded_process(command, log, seconds=protocol["execution"]["max_seconds_per_branch"] * branches, memory_mb=protocol["execution"]["memory_mb"], cpu_percent=protocol["execution"]["cpu_percent"], cuda=uses_cuda(pipeline), backend=protocol.get("resource_backend", "systemd"))
                record.update(outcome)
                if outcome["returncode"] != 0:
                    raise RuntimeError(f"execution failed; inspect {log.relative_to(root)}")
                validate_pipeline_bundle(directory / f"attempt-{attempt}", pipeline.canonical_payload())
                record.update(status="complete", reason=None)
            except (RuntimeError, ValueError, FileNotFoundError, subprocess.TimeoutExpired) as error:
                record.update(status="failed", reason=str(error))
                failures.append({"variant": row["name"], "curve": curve, "reason": str(error)})
            write_json_mapping_if_same(directory / f"outcome-{attempt}.json", record)
    return {"status": "execution_failed" if failures else "scores_ready", "failures": failures}


def seal_scores(root: Path) -> dict[str, Any]:
    protocol = verify_closed_plans(root)
    scores = {}
    for row in variants(protocol):
        for curve in planning_units(protocol):
            directory = root / "execution" / row["name"] / curve
            outcomes = sorted(directory.glob("outcome-*.json"))
            if not outcomes:
                raise ValueError("cannot freeze scores: missing target execution")
            outcome = read_json_mapping(outcomes[-1])
            if outcome["status"] != "complete":
                raise ValueError("cannot freeze scores: incomplete target/branch; no dropping curves")
            attempt = directory / f"attempt-{outcome['attempt']}"
            _, pipeline = resolved_job(root, protocol, row["name"], curve)
            validate_pipeline_bundle(attempt, pipeline.canonical_payload())
            task = task_for(protocol, curve)
            members = task.unit.held_out_curves if task.schema == "zero_shot" else task.unit.training_curves
            for member in members:
                source = attempt / "Results" / "Scores" / "TimeSigma" / protocol["schema"] / protocol["dataset"] / f"{member}.npy"
                values = np.load(source, allow_pickle=False)
                if values.ndim != 1 or not values.size or not np.isfinite(values).all():
                    raise ValueError("invalid final score")
                destination = root / "Results" / "Scores" / row["name"] / protocol["schema"] / protocol["dataset"] / f"{member}.npy"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copyfile(source, destination)
                if file_digest(destination) != file_digest(source):
                    raise ValueError("published score differs from frozen Pipeline output")
                scores[str(destination.relative_to(root))] = {"sha256": file_digest(destination), "source": str(source.relative_to(root)), "count": len(values)}
    record = {"planning_closed_sha256": file_digest(root / "planning-closed.json"), "scores": scores, "status": "scores_frozen"}
    write_json_mapping_if_same(root / "scores-frozen.json", record)
    return record


def evaluate_all(root: Path) -> None:
    from EasyTSAD.Controller import TSADController

    protocol = verify_closed_plans(root)
    if (root / "evaluation-complete.json").exists():
        raise ValueError("official evaluation is complete; regenerate the leaderboard without reevaluating")
    frozen = read_json_mapping(root / "scores-frozen.json")
    expected = {f"Results/Scores/{row['name']}/{protocol['schema']}/{protocol['dataset']}/{curve}.npy" for row in variants(protocol) for curve in evaluation_curves(protocol)}
    if set(frozen["scores"]) != expected or frozen["planning_closed_sha256"] != file_digest(root / "planning-closed.json"):
        raise ValueError("evaluation requires every score and every plan to be frozen")
    for name, details in frozen["scores"].items():
        if file_digest(root / name) != details["sha256"]:
            raise ValueError("score changed after freeze")
    with _working_directory(root.resolve()):
        controller = TSADController(log_level="error")
        controller.set_dataset(datasets=protocol["dataset"], dirname=protocol["dataset_root"], dataset_type="UTS", specify_curves=True, curve_names=evaluation_curves(protocol))
        controller.set_evals(official_evaluations())
        for row in variants(protocol):
            controller.do_evals(method=row["name"], training_schema=protocol["schema"])
        controller.summary.to_csv(datasets=[protocol["dataset"]], methods=[row["name"] for row in variants(protocol)], training_schema=protocol["schema"], eval_items=[list(item) for item in SUMMARY_ITEMS])


def leaderboard(root: Path) -> dict[str, Any]:
    protocol = verify_closed_plans(root)
    rows = []
    for row in variants(protocol):
        metrics = None
        failures = []
        coverage = []
        executions = 0
        reserved = 0
        wall = 0.0
        per_curve = {}
        for curve in planning_units(protocol):
            directory = root / "execution" / row["name"] / curve
            outcomes = sorted(directory.glob("outcome-*.json"))
            for path in outcomes:
                outcome = read_json_mapping(path)
                reserved += outcome["reserved_detector_executions"]
                wall += outcome.get("wall_seconds", 0)
                executions += len(list((directory / f"attempt-{outcome['attempt']}" / "branches").glob("*/run.json")))
            if outcomes and read_json_mapping(outcomes[-1])["status"] == "complete":
                task = task_for(protocol, curve)
                coverage.extend(task.unit.held_out_curves if task.schema == "zero_shot" else task.unit.training_curves)
            else:
                failures.append({"curve": curve, "reason": read_json_mapping(outcomes[-1])["reason"] if outcomes else "not executed"})
        evaluation = root / "Results" / "Evals" / row["name"] / protocol["schema"] / protocol["dataset"]
        if (root / "evaluation-complete.json").exists() and not failures:
            raw = read_json_mapping(evaluation / "all.json")
            if set(raw) != set(evaluation_curves(protocol)):
                failures.append({"reason": "official evaluator omitted a curve (including possible all-normal labels); no subset ranking"})
            else:
                metrics = _official_metrics(protocol["dataset"], read_json_mapping(evaluation / "avg.json"))
                per_curve = {curve: _official_metrics(protocol["dataset"], raw[curve]) for curve in evaluation_curves(protocol)}
        rows.append({"method": row["name"], "lifecycle": row.get("lifecycle", "planned_catalog_selection"), "status": "complete" if metrics is not None else "incomplete", "official_metrics": metrics, "per_curve": per_curve, "coverage": len(coverage), "required_curves": len(evaluation_curves(protocol)), "failures": failures, "detector_executions": executions, "reserved_detector_executions": reserved, "execution_wall_seconds": wall})
    primary = protocol["primary_metric"]
    complete = all(row["status"] == "complete" for row in rows)
    rows.sort(key=lambda row: (-(row["official_metrics"][primary] if row["official_metrics"] else -1), row["method"]))
    strongest = max((row["official_metrics"][primary] for row in rows if row["method"] != protocol["variant"] and row["official_metrics"]), default=None)
    for row in rows:
        value = row["official_metrics"][primary] if row["official_metrics"] else None
        row["rank"] = 1 + sum(other["official_metrics"][primary] > value for other in rows) if complete and protocol["baselines"] else None
        row["delta_vs_strongest_fixed_baseline"] = value - strongest if complete and strongest is not None else None
    usage = [read_json_mapping(root / "plans" / curve / "provider-outcome.json") for curve in planning_units(protocol)]
    report = {"status": "complete" if complete else "incomplete_no_ranking", "dataset": protocol["dataset"], "schema": protocol["schema"], "primary_metric": primary, "aggregation": "EasyTSAD official avg.json; same frozen score for all eight metrics", "rows": rows, "provider": {"calls": sum(x["usage"]["providerAttempts"] for x in usage), "tokens": sum(x["usage"]["tokens"] for x in usage), "wall_seconds": sum(x["wall_seconds"] for x in usage)}, "claim_scope": "frozen local EasyTSAD benchmark; no public SOTA claim"}
    if protocol.get("comparison"):
        report["claim_scope"] = "Local TimeSigma evaluation only; comparison ranks require the frozen published snapshot merge. Not an official submission."
    evaluation_attempts = [read_json_mapping(path) for path in sorted(root.glob("evaluation-attempt-*.json"))]
    report["evaluation"] = {"attempts": len(evaluation_attempts), "wall_seconds": sum(x["wall_seconds"] for x in evaluation_attempts)}
    # A derived view can be regenerated after a declared engineering repair.
    import json
    (root / "leaderboard.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (root / "leaderboard.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["method", "rank", "status", "coverage", *OFFICIAL_METRIC_IDS, "delta_vs_strongest_fixed_baseline", "detector_executions", "execution_wall_seconds"])
        for row in rows:
            writer.writerow([row["method"], row["rank"], row["status"], row["coverage"], *((row["official_metrics"] or {}).get(metric) for metric in OFFICIAL_METRIC_IDS), row["delta_vs_strongest_fixed_baseline"], row["detector_executions"], row["execution_wall_seconds"]])
    return report
