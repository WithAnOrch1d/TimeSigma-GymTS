from __future__ import annotations

import argparse
import json
from importlib.util import find_spec
from pathlib import Path
import random
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np
from packaging.specifiers import SpecifierSet
from packaging.version import Version
import toml

from timesigma.algorithms import EASYTSAD_SCHEMAS, get_method
from timesigma.algorithms.configuration import (
    freeze_configuration,
    resolve_configuration,
)
from timesigma.benchmark.manifest import build_run_manifest, write_run_manifest
from timesigma.benchmark.runner import run_easytsad, run_easytsad_scores


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_method(
    *,
    method_name: str,
    schema: str,
    datasets: Sequence[str],
    dataset_root: Path,
    dataset_source_root: Path | None = None,
    output_dir: Path,
    configuration_name: str = "default",
    configuration_path: Path | None = None,
    zero_shot_membership: Path | None = None,
    curve_names: Sequence[str] | None = None,
    seed: int = 1,
    model_artifacts: bool = False,
    evaluate: bool = True,
    label_free: bool = False,
    ordered_members: Mapping[str, Sequence[str]] | None = None,
) -> tuple[Path, Path]:
    spec = get_method(method_name)
    if method_name == "IsolationForest" and (type(seed) is not int or seed != 1):
        raise ValueError(f"{method_name} requires canonical seed=1")
    capability = spec.capability(schema)
    if not capability.supported:
        raise ValueError(
            f"{method_name} is N/A for {schema}: {capability.unavailable_reason}"
        )
    if zero_shot_membership is not None and schema != "zero_shot":
        raise ValueError("a frozen zero-shot membership requires schema zero_shot")
    curve_names = _naive_curve_filter(schema, datasets, curve_names)
    if ordered_members is not None and (schema != "all_in_one" or set(ordered_members) != set(datasets)):
        raise ValueError("ordered members require the complete ALL_IN_ONE dataset scope")
    configuration_path = (
        configuration_path.resolve() if configuration_path is not None else None
    )
    runtime_python = Path(sys.executable).absolute()
    python_version = Version(".".join(str(part) for part in sys.version_info[:3]))
    shared_runtime_python = None
    if spec.shared_runtime is not None:
        shared_runtime_python = (
            PROJECT_ROOT / ".runtimes" / spec.shared_runtime / "bin/python"
        ).absolute()

    if shared_runtime_python is None or runtime_python == shared_runtime_python:
        if python_version not in SpecifierSet(spec.python_requires):
            raise RuntimeError(
                f"{method_name} requires Python {spec.python_requires}; "
                f"current interpreter is Python {python_version}"
            )
        output_dir = output_dir.resolve()
        results_dir = output_dir / "Results"
        if results_dir.exists():
            raise FileExistsError(
                f"EasyTSAD results already exist: {results_dir}; "
                "use a new output directory"
            )
        overrides = (
            toml.load(configuration_path)
            if configuration_path is not None
            else None
        )
        resolved_configuration = resolve_configuration(
            spec.module,
            datasets,
            overrides,
        )
        defaults = resolve_configuration(spec.module, datasets)
        for dataset in datasets:
            spec.validate_parameters(resolved_configuration[dataset]["Model_Params"], defaults[dataset]["Model_Params"])
        frozen_configuration = freeze_configuration(
            output_dir / "configuration.toml",
            resolved_configuration,
            **({"pipeline": overrides["Pipeline"]} if overrides is not None and "Pipeline" in overrides else {}),
        )
        manifest_path = output_dir / "run.json"
        manifest = build_run_manifest(
            method_name=spec.method_name,
            module=spec.module,
            schema=schema,
            datasets=datasets,
            seed=seed,
            dataset_root=dataset_root,
            dataset_source_root=dataset_source_root,
            runtime_python=runtime_python,
            model_artifacts=model_artifacts,
            configuration_name=configuration_name,
            configuration_path=frozen_configuration,
        )
        if overrides is not None and "Pipeline" in overrides:
            manifest["pipeline_inputs"] = overrides["Pipeline"]
        if ordered_members is not None:
            manifest["ordered_members"] = {name: list(curves) for name, curves in ordered_members.items()}

        if label_free and evaluate:
            raise ValueError("label-free execution must defer evaluation")
        manifest["evaluation_deferred"] = not evaluate
        manifest["label_free_execution"] = True
        write_run_manifest(manifest_path, manifest)
        seed_runtime(seed)
        execute = run_easytsad if evaluate else run_easytsad_scores
        results_dir = execute(
            methods={spec.method_name: spec.module},
            schema=schema,
            datasets=datasets,
            dataset_root=dataset_root,
            output_dir=output_dir,
            configuration_paths={spec.method_name: frozen_configuration},
            zero_shot_membership=zero_shot_membership,
            curve_names=curve_names,
            **({"ordered_members": ordered_members} if ordered_members is not None else {}),
            artifact_dir=(
                output_dir / "Results" / "ModelArtifacts"
                if model_artifacts
                else None
            ),
        )
        manifest["status"] = "complete"
        write_run_manifest(manifest_path, manifest)
        return results_dir, runtime_python

    runtime_python = shared_runtime_python
    if not runtime_python.is_file():
        raise RuntimeError(
            f"{method_name} requires shared runtime {spec.shared_runtime}: "
            f"{runtime_python}"
        )

    command = [
        str(runtime_python),
        "-I",
        "-m",
        "timesigma.benchmark.runtime",
        "--method",
        method_name,
        "--schema",
        schema,
        "--dataset-root",
        str(dataset_root.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--configuration-name",
        configuration_name,
        "--seed",
        str(seed),
    ]
    if dataset_source_root is not None:
        command.extend(
            ("--dataset-source-root", str(dataset_source_root.resolve()))
        )
    if configuration_path is not None:
        command.extend(("--configuration", str(configuration_path)))
    for dataset in datasets:
        command.extend(("--dataset", dataset))
    if zero_shot_membership is not None:
        command.extend(
            ("--zero-shot-membership", str(zero_shot_membership.resolve()))
        )
    if curve_names is not None:
        for curve in curve_names:
            command.extend(("--curve", curve))
    if ordered_members is not None:
        command.extend(("--ordered-members-json", json.dumps({name: list(curves) for name, curves in ordered_members.items()})))
    if model_artifacts:
        command.append("--model-artifacts")
    if not evaluate:
        command.append("--scores-only")
    if label_free:
        command.append("--label-free")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    return output_dir.resolve() / "Results", runtime_python


def seed_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if find_spec("torch") is None:
        return

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _naive_curve_filter(
    schema: str,
    datasets: Sequence[str],
    curve_names: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if curve_names is None:
        return None
    if isinstance(curve_names, (str, bytes)):
        raise TypeError("curve_names must be a sequence of curve names")
    curves = tuple(curve_names)
    if schema != "naive":
        raise ValueError("curve_names is only valid for schema naive")
    if len(datasets) != 1 or len(curves) != 1:
        raise ValueError("a naive curve filter requires one dataset and one curve")
    if not isinstance(curves[0], str) or not curves[0].strip():
        raise TypeError("curve_names must contain one non-empty string")
    return curves


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--schema", required=True, choices=EASYTSAD_SCHEMAS)
    parser.add_argument("--dataset", dest="datasets", action="append", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--configuration-name", default="default")
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--zero-shot-membership", type=Path)
    parser.add_argument("--curve", dest="curve_names", action="append")
    parser.add_argument("--model-artifacts", action="store_true")
    parser.add_argument("--scores-only", action="store_true")
    parser.add_argument("--label-free", action="store_true", help="Require scores-only execution; detector labels are always isolated")
    parser.add_argument("--ordered-members-json", type=json.loads)
    args = parser.parse_args(argv)
    run_method(
        method_name=args.method,
        schema=args.schema,
        datasets=tuple(args.datasets),
        dataset_root=args.dataset_root,
        dataset_source_root=args.dataset_source_root,
        output_dir=args.output_dir,
        configuration_name=args.configuration_name,
        configuration_path=args.configuration,
        zero_shot_membership=args.zero_shot_membership,
        curve_names=args.curve_names,
        seed=args.seed,
        model_artifacts=args.model_artifacts,
        evaluate=not args.scores_only,
        label_free=args.label_free,
        ordered_members=args.ordered_members_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["run_method"]
