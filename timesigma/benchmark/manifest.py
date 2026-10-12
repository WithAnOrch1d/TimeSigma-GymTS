from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import toml

from timesigma.algorithms import EASYTSAD_REVISION
from timesigma.algorithms.configuration import resolve_configuration
from timesigma.benchmark.zero_shot import MEMBERSHIP_FILENAME


def build_run_manifest(
    *,
    method_name: str,
    module: str,
    schema: str,
    datasets: Sequence[str],
    seed: int,
    dataset_root: Path,
    dataset_source_root: Path | None = None,
    runtime_python: Path,
    model_artifacts: bool = False,
    configuration_name: str,
    configuration_path: Path,
) -> dict[str, object]:
    dataset_root = dataset_root.resolve()
    dataset_source_root = (
        dataset_root
        if dataset_source_root is None
        else dataset_source_root.resolve()
    )
    configuration_path = configuration_path.resolve()
    config = toml.load(configuration_path)
    resolved = resolve_configuration(module, datasets, config)

    manifest: dict[str, object] = {
        "status": "running",
        "method": method_name,
        "module": module,
        "schema": schema,
        "configuration_name": configuration_name,
        "configuration_path": configuration_path.name,
        "datasets": list(datasets),
        "seed": seed,
        "easytsad_revision": EASYTSAD_REVISION,
        "dataset": dataset_provenance(dataset_source_root),
        "runtime_python": str(runtime_python.absolute()),
        "config": {
            "Data_Params": config["Data_Params"],
            "Model_Params": config["Model_Params"],
            "Resolved": resolved,
        },
        "results_path": "Results",
        "zero_shot_membership_path": (
            f"Results/{MEMBERSHIP_FILENAME}" if schema == "zero_shot" else None
        ),
    }
    if model_artifacts:
        manifest["model_artifacts"] = {"path": "Results/ModelArtifacts"}
    return manifest


def write_run_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_value(dataset_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(dataset_root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def dataset_provenance(dataset_root: Path) -> dict[str, str | bool | None]:
    dataset_root = dataset_root.resolve()
    if not (dataset_root / ".git").exists():
        return {
            "root": str(dataset_root),
            "source": None,
            "revision": None,
            "clean": None,
        }
    source = _git_value(dataset_root, "config", "--get", "remote.origin.url")
    return {
        "root": str(dataset_root),
        "source": _sanitize_git_source(source),
        "revision": _git_value(dataset_root, "rev-parse", "HEAD"),
        "clean": _git_value(dataset_root, "status", "--short") == "",
    }


def _sanitize_git_source(source: str) -> str:
    parsed = urlsplit(source)
    if parsed.scheme not in {"http", "https"} or "@" not in parsed.netloc:
        return source
    netloc = parsed.netloc.rsplit("@", 1)[1]
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


__all__ = [
    "build_run_manifest",
    "dataset_provenance",
    "write_run_manifest",
]
