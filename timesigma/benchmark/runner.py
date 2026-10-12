from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from timesigma.algorithms.imports import import_method
import os
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import toml

from EasyTSAD.Controller import TSADController
from EasyTSAD.Evaluations import EvalInterface
from EasyTSAD.Evaluations.Protocols import (
    EventF1PA,
    EventKthF1PA,
    EventKthPrcPA,
    EventPrcPA,
    PointAuprcPA,
    PointF1PA,
    PointKthF1PA,
)

from timesigma.algorithms.catalog import EASYTSAD_SCHEMAS, METHOD_CATALOG, get_method
from timesigma.algorithms.configuration import resolve_configuration
from timesigma.benchmark.zero_shot import (
    bind_zero_shot_membership,
    freeze_zero_shot_membership,
    validate_zero_shot_scores,
)

DELAY_VALUES = (3, 10, 20, 50, 150)
LEADERBOARD_DELAY = {
    "AIOPS": 10,
    "NAB": 150,
    "TODS": 3,
    "WSD": 10,
    "Yahoo": 3,
    "UCR": 50,
    "NEK": 10,
}

SUMMARY_ITEMS = (
    ("best f1 under pa", "f1"),
    ("event-based f1 under pa with mode log", "f1"),
    ("event-based f1 under pa with mode squeeze", "f1"),
    *((f"event-based f1 under {delay}-delay pa with mode log", "f1") for delay in DELAY_VALUES),
    ("point-based auprc pa",),
    ("event-based auprc under pa with mode log",),
    ("event-based auprc under pa with mode squeeze",),
    *((f"{delay}-th auprc under event-based pa with mode log",) for delay in DELAY_VALUES),
)


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def official_evaluations() -> list[EvalInterface]:
    evaluations: list[EvalInterface] = [
        PointF1PA(),
        EventF1PA(),
        EventF1PA(mode="squeeze"),
    ]
    evaluations.extend(PointKthF1PA(delay) for delay in DELAY_VALUES)
    evaluations.append(PointAuprcPA())
    evaluations.extend(EventKthPrcPA(delay, mode="raw") for delay in DELAY_VALUES)
    evaluations.extend(
        EventKthF1PA(delay, mode="log", base=3) for delay in DELAY_VALUES
    )
    evaluations.append(EventPrcPA(mode="log", base=3))
    evaluations.extend(
        EventKthPrcPA(delay, mode="log", base=3) for delay in DELAY_VALUES
    )
    evaluations.extend(EventKthF1PA(delay, mode="squeeze") for delay in DELAY_VALUES)
    evaluations.append(EventPrcPA(mode="squeeze"))
    evaluations.extend(
        EventKthPrcPA(delay, mode="squeeze") for delay in DELAY_VALUES
    )
    return evaluations


def leaderboard_items(dataset: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    delay = LEADERBOARD_DELAY[dataset]
    return (
        ("point-wise F1", ("best f1 under pa", "f1")),
        (
            "reduced-length F1",
            ("event-based f1 under pa with mode log", "f1"),
        ),
        (
            "event-wise F1",
            ("event-based f1 under pa with mode squeeze", "f1"),
        ),
        (
            "delay reduced-length F1",
            (f"event-based f1 under {delay}-delay pa with mode log", "f1"),
        ),
        ("point-wise AUPRC", ("point-based auprc pa",)),
        (
            "reduced-length AUPRC",
            ("event-based auprc under pa with mode log",),
        ),
        (
            "event-wise AUPRC",
            ("event-based auprc under pa with mode squeeze",),
        ),
        (
            "delay reduced-length AUPRC",
            (f"{delay}-th auprc under event-based pa with mode log",),
        ),
    )


def run_easytsad(
    *,
    methods: Mapping[str, str],
    schema: str,
    datasets: Sequence[str],
    dataset_root: Path,
    output_dir: Path,
    hparams: Mapping[str, Mapping[str, object]] | None = None,
    configuration_paths: Mapping[str, Path] | None = None,
    zero_shot_membership: Path | None = None,
    curve_names: Sequence[str] | None = None,
    artifact_dir: Path | None = None,
    ordered_members: Mapping[str, Sequence[str]] | None = None,
) -> Path:
    controller, results_dir = _run_scores(
        methods=methods,
        schema=schema,
        datasets=datasets,
        dataset_root=dataset_root,
        output_dir=output_dir,
        hparams=hparams,
        configuration_paths=configuration_paths,
        zero_shot_membership=zero_shot_membership,
        curve_names=curve_names,
        artifact_dir=artifact_dir,
        ordered_members=ordered_members,
    )
    with _working_directory(output_dir.resolve()):
        controller.set_evals(official_evaluations())
        for method in methods:
            controller.do_evals(method=method, training_schema=schema)

        controller.summary.to_csv(
            datasets=list(datasets),
            methods=list(methods),
            training_schema=schema,
            eval_items=[list(item) for item in SUMMARY_ITEMS],
        )

    return results_dir


def run_easytsad_scores(
    *,
    methods: Mapping[str, str],
    schema: str,
    datasets: Sequence[str],
    dataset_root: Path,
    output_dir: Path,
    hparams: Mapping[str, Mapping[str, object]] | None = None,
    configuration_paths: Mapping[str, Path] | None = None,
    zero_shot_membership: Path | None = None,
    curve_names: Sequence[str] | None = None,
    artifact_dir: Path | None = None,
    ordered_members: Mapping[str, Sequence[str]] | None = None,
) -> Path:
    _, results_dir = _run_scores(
        methods=methods,
        schema=schema,
        datasets=datasets,
        dataset_root=dataset_root,
        output_dir=output_dir,
        hparams=hparams,
        configuration_paths=configuration_paths,
        zero_shot_membership=zero_shot_membership,
        curve_names=curve_names,
        artifact_dir=artifact_dir,
        ordered_members=ordered_members,
    )
    return results_dir


def _run_scores(
    *,
    methods: Mapping[str, str],
    schema: str,
    datasets: Sequence[str],
    dataset_root: Path,
    output_dir: Path,
    hparams: Mapping[str, Mapping[str, object]] | None,
    configuration_paths: Mapping[str, Path] | None,
    zero_shot_membership: Path | None,
    curve_names: Sequence[str] | None,
    artifact_dir: Path | None,
    ordered_members: Mapping[str, Sequence[str]] | None = None,
) -> tuple[TSADController, Path]:
    if schema not in EASYTSAD_SCHEMAS:
        raise ValueError(f"unsupported EasyTSAD schema: {schema}")
    if not methods or not datasets:
        raise ValueError("at least one method and dataset are required")
    if zero_shot_membership is not None and schema != "zero_shot":
        raise ValueError("a frozen zero-shot membership requires schema zero_shot")
    curve_names = _naive_curve_filter(schema, datasets, curve_names)
    if ordered_members is not None and (schema != "all_in_one" or set(ordered_members) != set(datasets)):
        raise ValueError("ordered members require the complete ALL_IN_ONE dataset scope")

    resolved_configuration_paths = {
        method: path.resolve()
        for method, path in (configuration_paths or {}).items()
    }
    unknown_configuration_methods = set(resolved_configuration_paths) - set(methods)
    if unknown_configuration_methods:
        unknown = ", ".join(sorted(unknown_configuration_methods))
        raise ValueError(f"configuration supplied for an unknown method: {unknown}")
    for method in resolved_configuration_paths:
        if hparams is not None and method in hparams:
            raise ValueError(
                f"{method} cannot use configuration_path and hparams together"
            )
    # External library adapters keep their declared config in this repository;
    # EasyTSAD's default path lookup only knows its own site-packages Methods.
    for external in ("IsolationForest",):
        if external in methods and external not in resolved_configuration_paths:
            from timesigma.algorithms.configuration import default_configuration_path

            if hparams is not None and external in hparams:
                spec = get_method(external)
                defaults = resolve_configuration(spec.module, datasets)
                for dataset in datasets:
                    spec.validate_parameters(hparams[external], defaults[dataset]["Model_Params"])
            resolved_configuration_paths[external] = default_configuration_path(methods[external])

    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    if zero_shot_membership is not None:
        zero_shot_membership = zero_shot_membership.resolve()
    artifact_dir = artifact_dir.resolve() if artifact_dir is not None else None
    results_dir = output_dir / "Results"
    if results_dir.exists():
        raise FileExistsError(
            f"EasyTSAD results already exist: {results_dir}; use a new output directory"
        )

    data_params = {
        method: _timesigma_data_params(schema, hparams) if method == "TimeSigma" else {}
        for method in methods
    }

    for module in methods.values():
        import_method(module)
    configurations = {}
    for method, module in methods.items():
        if method not in METHOD_CATALOG and artifact_dir is None:
            continue
        configuration_path = resolved_configuration_paths.get(method)
        overrides = toml.load(configuration_path) if configuration_path is not None else None
        configurations[method] = resolve_configuration(module, datasets, overrides)
        defaults = resolve_configuration(module, datasets)
        method_hparams = None if hparams is None else hparams.get(method)
        for dataset in datasets:
            if configuration_path is None and method_hparams is not None:
                configurations[method][dataset]["Model_Params"] = deepcopy(method_hparams)
            get_method(method).validate_parameters(
                configurations[method][dataset]["Model_Params"], defaults[dataset]["Model_Params"],
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with _working_directory(output_dir), bind_ordered_members(ordered_members):
        controller = TSADController(log_level="error")
        dataset_scope = (
            {}
            if curve_names is None
            else {"specify_curves": True, "curve_names": list(curve_names)}
        )
        controller.set_dataset(
            datasets=(datasets[0] if curve_names is not None else list(datasets)),
            dirname=str(dataset_root),
            dataset_type="UTS",
            **dataset_scope,
        )
        splits = None
        if schema == "zero_shot":
            splits = freeze_zero_shot_membership(
                controller,
                datasets,
                results_dir,
                expected_path=zero_shot_membership,
            )
        for method in methods:
            configuration_path = resolved_configuration_paths.get(method)
            artifacts = nullcontext()
            inputs = nullcontext()
            admission = nullcontext()
            execution = nullcontext()
            from timesigma.benchmark.label_free import label_free_loading

            label_boundary = label_free_loading()
            if method in ("DCdetector", "FCVAE"):
                import torch

                execution = torch.backends.cudnn.flags(
                    enabled=torch.backends.cudnn.enabled, benchmark=False,
                    deterministic=True, allow_tf32=torch.backends.cudnn.allow_tf32,
                )
            if configuration_path is not None:
                pipeline = toml.load(configuration_path).get("Pipeline")
                if pipeline is not None:
                    from timesigma.benchmark.pipeline_inputs import bind_pipeline_inputs

                    inputs = bind_pipeline_inputs(configuration=pipeline, schema=schema, directory=output_dir, splits=splits)
            if method in ("LSTMADalpha", "LSTMADbeta", "FITS", "DCdetector", "FCVAE", "IsolationForest"):
                from timesigma.benchmark.input_contract import bind_input_contract

                admission = bind_input_contract(get_method(method), schema, configurations[method], splits)
            if artifact_dir is not None:
                from timesigma.benchmark.artifacts import bind_model_artifacts

                artifacts = bind_model_artifacts(
                    method=method,
                    schema=schema,
                    directory=artifact_dir,
                    configurations=configurations[method],
                    zero_shot_splits=splits,
                )
            with execution, label_boundary, bind_zero_shot_membership(splits), inputs, admission as score_support, artifacts:
                controller.run_exps(
                    method=method,
                    training_schema=schema,
                    cfg_path=(
                        str(configuration_path)
                        if configuration_path is not None
                        else None
                    ),
                    preprocess=(
                        None
                        if configuration_path is not None
                        else data_params[method].get("preprocess")
                    ),
                    diff_order=(
                        None
                        if configuration_path is not None
                        else data_params[method].get("diff_order")
                    ),
                    hparams=(
                        None
                        if configuration_path is not None or hparams is None
                        else hparams.get(method)
                    ),
                )
            if score_support is not None:
                from timesigma.benchmark.input_contract import validate_score_support

                validate_score_support(output_dir, method, schema, score_support)
            if splits is not None:
                validate_zero_shot_scores(results_dir, (method,), splits)

    return controller, results_dir


@contextmanager
def bind_ordered_members(members: Mapping[str, Sequence[str]] | None) -> Iterator[None]:
    """Use the caller's frozen task order in the native loader, without subsetting."""
    if members is None:
        yield
        return
    from EasyTSAD.Controller.PathManager import PathManager

    frozen = {}
    for dataset, curves in members.items():
        if isinstance(curves, (str, bytes)) or not curves or any(not isinstance(c, str) or not c for c in curves):
            raise ValueError("ordered members must contain non-empty curve sequences")
        if len(set(curves)) != len(curves):
            raise ValueError("ordered members must be unique")
        frozen[dataset] = tuple(curves)
    original = PathManager.get_dataset_curves

    def get_curves(manager, kind, dataset):
        actual = original(manager, kind, dataset)
        if dataset not in frozen:
            return actual
        expected = frozen[dataset]
        if len(actual) != len(expected) or set(actual) != set(expected):
            raise ValueError("native dataset members differ from the complete frozen task")
        return list(expected)

    PathManager.get_dataset_curves = get_curves
    try:
        yield
    finally:
        PathManager.get_dataset_curves = original


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


def _timesigma_data_params(
    schema: str,
    hparams: Mapping[str, Mapping[str, object]] | None,
) -> Mapping[str, object]:
    params = None if hparams is None else hparams.get("TimeSigma")
    if params is None:
        raise ValueError("TimeSigma hparams are required")
    if params.get("schema") != schema:
        raise ValueError("TimeSigma schema must match the Controller training schema")

    component = params.get("component")
    if not isinstance(component, str):
        raise TypeError("TimeSigma component must be a string")
    module = import_method(get_method(component).module)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise RuntimeError(f"EasyTSAD component module has no source path: {component}")
    config_path = Path(module_file).with_name("config.toml")
    if not config_path.is_file():
        raise FileNotFoundError(f"EasyTSAD component config is missing: {config_path}")
    config = toml.load(config_path)
    data_params = config.get("Data_Params")
    if not isinstance(data_params, Mapping):
        raise ValueError(f"EasyTSAD component config has no Data_Params: {component}")
    return data_params
