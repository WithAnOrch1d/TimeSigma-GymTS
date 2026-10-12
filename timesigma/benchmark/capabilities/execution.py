from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import toml

from timesigma.algorithms import EASYTSAD_REVISION, EASYTSAD_SCHEMAS, METHOD_CATALOG

from .. import runtime
from ..leaderboard import OFFICIAL_METRIC_IDS
from ..zero_shot import MEMBERSHIP_FILENAME
from . import sources
from .validation import validate_capability_job


def run_capabilities(
    plan_path: Path,
    dataset_root: Path,
    output_dir: Path,
    source_roots: Sequence[Path] = (),
    methods: Sequence[str] = (),
    schemas: Sequence[str] = (),
    datasets: Sequence[str] = (),
) -> dict[str, object]:
    plan_path = plan_path.resolve()
    dataset_root = dataset_root.resolve()
    source_roots = tuple(path.resolve() for path in source_roots)
    output_dir = output_dir.resolve()
    paths = [
        ("plan", plan_path),
        ("dataset root", dataset_root),
        ("output", output_dir),
    ]
    paths.extend(("source root", path) for path in source_roots)
    for label, path in paths:
        _reject_aiops_path(label, path)

    plan = _load_plan(plan_path)
    selected_methods = _select(plan["methods"], methods, "method")
    selected_schemas = _select(plan["schemas"], schemas, "schema")
    selected_datasets = _select(plan["datasets"], datasets, "dataset")
    dataset_sources = sources.resolve_dataset_sources(
        plan, dataset_root, source_roots, selected_datasets
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, object]] = []
    configuration_name = str(plan["configuration"])
    with sources.dataset_view(dataset_sources) as easytsad_root:
        for method in selected_methods:
            spec = METHOD_CATALOG[method]
            for schema in selected_schemas:
                capability = spec.capability(schema)
                for dataset in selected_datasets:
                    source = dataset_sources[dataset]
                    final_dir = (
                        output_dir / method / configuration_name / schema / dataset
                    )
                    if not capability.supported:
                        jobs.append(
                            _job(
                                method,
                                configuration_name,
                                schema,
                                dataset,
                                "N/A",
                                final_dir,
                            )
                        )
                        continue
                    expected_run = {
                        "status": "complete",
                        "method": method,
                        "module": spec.module,
                        "schema": schema,
                        "configuration_name": configuration_name,
                        "configuration_path": "configuration.toml",
                        "datasets": [dataset],
                        "seed": int(plan["seed"]),
                        "easytsad_revision": plan["easytsad_revision"],
                        "dataset": dict(source.provenance),
                        "results_path": "Results",
                        "zero_shot_membership_path": (
                            f"Results/{MEMBERSHIP_FILENAME}"
                            if schema == "zero_shot"
                            else None
                        ),
                        "model_artifacts": {"path": "Results/ModelArtifacts"},
                    }
                    shared_split = output_dir / "zero_shot" / f"{dataset}.json"
                    expected_split = (
                        shared_split
                        if schema == "zero_shot" and shared_split.is_file()
                        else None
                    )
                    if final_dir.exists():
                        validate_capability_job(
                            final_dir, easytsad_root, expected_run, expected_split
                        )
                        if schema == "zero_shot" and expected_split is None:
                            _publish_membership(final_dir, shared_split)
                        jobs.append(
                            _job(
                                method,
                                configuration_name,
                                schema,
                                dataset,
                                "skipped",
                                final_dir,
                            )
                        )
                        continue

                    attempt_parent = (
                        output_dir
                        / ".attempts"
                        / method
                        / configuration_name
                        / schema
                        / dataset
                    )
                    attempt_parent.mkdir(parents=True, exist_ok=True)
                    attempt_dir = Path(
                        tempfile.mkdtemp(prefix="attempt-", dir=attempt_parent)
                    )
                    results_dir, _ = runtime.run_method(
                        method_name=method,
                        schema=schema,
                        datasets=(dataset,),
                        dataset_root=easytsad_root,
                        dataset_source_root=source.root,
                        output_dir=attempt_dir,
                        configuration_name=configuration_name,
                        zero_shot_membership=expected_split,
                        seed=int(plan["seed"]),
                        model_artifacts=True,
                    )
                    if results_dir.resolve() != (attempt_dir / "Results").resolve():
                        raise RuntimeError(
                            "EasyTSAD returned an unexpected Results directory"
                        )
                    validate_capability_job(
                        attempt_dir, easytsad_root, expected_run, expected_split
                    )
                    final_dir.parent.mkdir(parents=True, exist_ok=True)
                    attempt_dir.rename(final_dir)
                    if schema == "zero_shot" and expected_split is None:
                        _publish_membership(final_dir, shared_split)
                    jobs.append(
                        _job(
                            method,
                            configuration_name,
                            schema,
                            dataset,
                            "complete",
                            final_dir,
                        )
                    )

    counts = {
        status: sum(job["status"] == status for job in jobs)
        for status in ("complete", "skipped", "N/A")
    }
    return {
        "plan": str(plan_path),
        "dataset_source": plan["dataset_source"],
        "dataset_revision": plan["dataset_revision"],
        "seed": plan["seed"],
        "jobs": jobs,
        "counts": counts,
    }


def _load_plan(path: Path) -> dict[str, object]:
    payload = toml.load(path)
    for key in ("datasets", "methods", "schemas", "metrics"):
        values = payload[key]
        if not isinstance(values, list) or not all(
            isinstance(item, str) for item in values
        ):
            raise TypeError(f"capability plan {key} must be a list of strings")
        payload[key] = tuple(values)
    payload["seed"] = int(payload["seed"])
    _reject_aiops_names(payload["datasets"])
    if payload["metrics"] != OFFICIAL_METRIC_IDS:
        raise ValueError("capability plan must use the eight official metrics")
    if payload["configuration"] != "default":
        raise ValueError("capability plan must use the default configuration")
    if payload["easytsad_revision"] != EASYTSAD_REVISION:
        raise ValueError("capability plan must use the audited EasyTSAD revision")
    if payload["comparison_variable"] != "method":
        raise ValueError("capability comparison variable must be method")
    execution = (
        int(payload["repetitions"]),
        int(payload["parallel_jobs"]),
        payload["stop_on_error"],
    )
    if execution != (1, 1, True):
        raise ValueError("capability execution must use one serial fail-fast repetition")
    if set(payload["methods"]) - set(METHOD_CATALOG):
        raise ValueError("capability plan contains an unknown method")
    if set(payload["schemas"]) - set(EASYTSAD_SCHEMAS):
        raise ValueError("capability plan contains an unknown schema")
    payload["dataset_sources"] = _load_dataset_sources(payload)
    return payload


def _select(
    available: tuple[str, ...], requested: Sequence[str], label: str
) -> tuple[str, ...]:
    if not requested:
        return available
    requested_set = set(requested)
    if len(requested_set) != len(requested) or requested_set - set(available):
        raise ValueError(f"capability selection contains an invalid {label}")
    return tuple(item for item in available if item in requested_set)


def _publish_membership(directory: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(directory / "Results" / MEMBERSHIP_FILENAME, destination)


def _reject_aiops_names(datasets: Sequence[str]) -> None:
    if any(dataset.casefold() == "aiops" for dataset in datasets):
        raise ValueError("AIOPS is forbidden in capability experiments")


def _reject_aiops_path(label: str, path: Path) -> None:
    if any(part.casefold() == "aiops" for part in path.parts):
        raise ValueError(f"AIOPS is forbidden in the capability {label} path")


def _load_dataset_sources(plan: Mapping[str, object]) -> dict[str, Mapping[str, str]]:
    sources = plan.get("dataset_sources", {})
    if not isinstance(sources, Mapping):
        raise TypeError("capability dataset_sources must be a mapping")
    if set(sources) - set(plan["datasets"]):
        raise ValueError("capability dataset source names must be planned datasets")
    loaded: dict[str, Mapping[str, str]] = {}
    for dataset, source in sources.items():
        if not isinstance(dataset, str) or not isinstance(source, Mapping):
            raise TypeError("capability dataset sources must be named mappings")
        required = ("source", "revision", "relative_path")
        if not all(isinstance(source.get(key), str) and source[key] for key in required):
            raise TypeError(f"capability dataset source is incomplete: {dataset}")
        loaded[dataset] = {key: str(source[key]) for key in required}
    return loaded


def _job(
    method: str,
    configuration_name: str,
    schema: str,
    dataset: str,
    status: str,
    directory: Path,
) -> dict[str, object]:
    return {
        "method": method,
        "configuration": configuration_name,
        "schema": schema,
        "dataset": dataset,
        "status": status,
        "output_dir": str(directory),
    }


__all__ = ["run_capabilities"]
