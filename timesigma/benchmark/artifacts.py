from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from EasyTSAD.Controller import TSADController as _TSADController  # noqa: F401
from EasyTSAD.Methods import BaseMethodMeta
from EasyTSAD.TrainingSchema.BaseSchema import BaseSchema

from timesigma.algorithms import AlgorithmSpec, get_method
from timesigma.algorithms.artifacts import (
    resolve_selection_status,
    save_fitted_estimator,
    score_source_for_execution,
)
from timesigma.benchmark.zero_shot import ZeroShotSplit, split_for_dataset


@dataclass(frozen=True)
class _Identity:
    dataset: str
    curve: str


class _LoadedSeries:
    def __init__(self) -> None:
        self._identity: dict[int, _Identity] = {}
        self._datasets: dict[str, dict[str, Any]] = {}

    def record(self, loaded: Mapping[str, Mapping[str, Any]]) -> None:
        for dataset, curves in loaded.items():
            self._datasets[dataset] = dict(curves)
            for curve, series in curves.items():
                self._identity[id(series)] = _Identity(dataset, curve)

    def identify(self, series: Any) -> _Identity:
        try:
            return self._identity[id(series)]
        except KeyError as exc:
            raise RuntimeError("Model Artifact received an unobserved TSData instance") from exc

    def dataset(self, dataset: str) -> Mapping[str, Any]:
        return self._datasets[dataset]


@dataclass(frozen=True)
class _Unit:
    root: Path
    spec: AlgorithmSpec
    schema: str
    dataset: str
    curve: str | None
    training: Mapping[str, Any]
    testing: Mapping[str, Any]
    configuration: Mapping[str, object]
    split: ZeroShotSplit | None

    @property
    def directory(self) -> Path:
        path = self.root / self.spec.method_name / self.schema / self.dataset
        return path / self.curve if self.curve is not None else path

    def metadata(
        self,
        method: Any,
        *,
        artifact_path: str | None,
        selection_values: Sequence[float] | None,
        best_epoch: int | None,
        allow_pending_artifact: bool = False,
    ) -> dict[str, object]:
        capability = self.spec.capability(self.schema)
        counts = _sample_counts(self.spec, method, self.training, self.testing)
        if (
            self.spec.artifact_type != "replay_recipe"
            and counts["train_samples"] == 0
        ):
            raise ValueError(
                f"{self.spec.method_name}/{self.schema}/{self.dataset} has no training samples"
            )

        selection_status, selection_reason = resolve_selection_status(
            capability,
            counts,
        )

        if (
            selection_status == "available"
            and artifact_path is None
            and not allow_pending_artifact
        ):
            raise RuntimeError("available model selection has no persisted artifact")
        if selection_status != "available" and selection_values is not None:
            raise RuntimeError("selection values require available model selection")

        score_source = score_source_for_execution(
            selection_status,
            self.spec.artifact_type,
        )
        payload: dict[str, object] = {
            "method": self.spec.method_name,
            "schema": self.schema,
            "dataset": self.dataset,
            "curve": self.curve,
            "artifact_type": self.spec.artifact_type,
            "artifact_path": artifact_path,
            "selection_status": selection_status,
            "selection_policy": capability.selection_policy,
            "selection_objective": capability.selection_objective,
            "selection_values": (
                list(selection_values) if selection_values is not None else None
            ),
            "selection_reason": selection_reason,
            "best_epoch": best_epoch,
            "score_source": score_source,
            "training_curves": list(self.training),
            "test_curves": list(self.testing),
            "sample_counts": counts,
            "configuration": dict(self.configuration),
        }
        if self.split is not None:
            payload["zero_shot_split"] = self.split.as_dict()
        if self.spec.method_name == "IsolationForest":
            from timesigma.algorithms.isolation_forest.binding import (
                SOURCE_IDENTITY, fit_rows, runtime_binding, source_binding,
            )

            run = json.loads((self.root.parent.parent / "run.json").read_text())
            payload.update(
                seed=run["seed"], source_identity=SOURCE_IDENTITY,
                source_binding=source_binding(), runtime_binding=runtime_binding(),
                dataset_revision=run["dataset"]["revision"], easytsad_revision=run["easytsad_revision"],
                fit_rows=fit_rows(self.training), input_features=1,
                row_order="training_curves order, then ascending within-member timestamps",
                score_definition="-estimator.score_samples(X_test)",
                score_support={curve: self.spec.score_support(len(series.test), self.configuration["Model_Params"]) for curve, series in self.testing.items()},
            )
        if self.spec.method_name in ("LSTMADalpha", "LSTMADbeta", "FITS", "DCdetector", "FCVAE"):
            run = json.loads((self.root.parent.parent / "run.json").read_text())
            payload.update(
                seed=run["seed"], source_identity="timesigma-selected",
                score_support={curve: self.spec.score_support(len(series.test), self.configuration["Model_Params"]) for curve, series in self.testing.items()},
            )
        return payload

    def write(self, payload: Mapping[str, object]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _sample_counts(
    spec: AlgorithmSpec,
    method: Any,
    training: Mapping[str, Any],
    testing: Mapping[str, Any],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for phase in ("train", "valid"):
        lengths = [len(getattr(series, phase)) for series in training.values()]
        counts[f"{phase}_points"] = sum(lengths)
        counts[f"{phase}_samples"] = sum(
            spec.sample_spec.count(length, method) for length in lengths
        )

    test_lengths = [len(series.test) for series in testing.values()]
    test_samples = [spec.sample_spec.count(length, method) for length in test_lengths]
    if spec.method_name == "DCdetector":
        test_samples = [length // method.win_size for length in test_lengths]
    empty_tests = [curve for curve, samples in zip(testing, test_samples) if samples == 0]
    if empty_tests:
        raise ValueError(
            f"{spec.method_name} has no test samples for: {', '.join(empty_tests)}"
        )
    counts["test_points"] = sum(test_lengths)
    counts["test_samples"] = sum(test_samples)
    if spec.method_name == "IsolationForest":
        counts["fit_samples"] = counts["train_points"]
    elif spec.artifact_type == "fitted_estimator":
        counts["fit_samples"] = counts["train_points"] + counts["valid_points"]
    elif spec.artifact_type == "replay_recipe":
        counts["fit_samples"] = 0
    else:
        counts["fit_samples"] = counts["train_samples"]
    return counts


def _capture_unit(unit: _Unit, method: Any, train: Callable[[], None]) -> None:
    preflight = unit.metadata(
        method,
        artifact_path=None,
        selection_values=None,
        best_epoch=None,
        allow_pending_artifact=True,
    )
    status = preflight["selection_status"]

    if status == "available":
        if unit.spec.artifact_type != "torch_state_dict":
            raise TypeError(
                f"{unit.spec.method_name} has no selected-state adapter"
            )
        from timesigma.algorithms.checkpoint import SavedCheckpoint, retain_best

        saved: list[SavedCheckpoint] = []
        with retain_best(method, unit.directory / "best.pt", saved.append):
            train()
        selected = saved[0]
        if unit.spec.method_name == "FCVAE":
            method._fcvae_replay_destination = selected.path
        unit.write(
            unit.metadata(
                method,
                artifact_path="best.pt",
                selection_values=selected.values,
                best_epoch=selected.best_epoch,
            )
        )
        return

    train()
    artifact_path = None
    if unit.spec.artifact_type == "fitted_estimator":
        artifact_path = "estimator.pkl"
        if unit.spec.method_name == "IsolationForest":
            from timesigma.algorithms.isolation_forest.binding import estimator_binding

            method.estimator._timesigma_binding = estimator_binding(preflight)
        save_fitted_estimator(method, unit.spec, unit.directory / artifact_path)
    unit.write(
        unit.metadata(
            method,
            artifact_path=artifact_path,
            selection_values=None,
            best_epoch=None,
        )
    )


@contextmanager
def _observe_loaded_series() -> Iterator[_LoadedSeries]:
    loaded_series = _LoadedSeries()
    original = BaseSchema.load_data

    def load_data(schema: BaseSchema) -> Mapping[str, Mapping[str, Any]]:
        loaded = original(schema)
        loaded_series.record(loaded)
        return loaded

    BaseSchema.load_data = load_data
    try:
        yield loaded_series
    finally:
        BaseSchema.load_data = original


@contextmanager
def bind_model_artifacts(
    *,
    method: str,
    schema: str,
    directory: Path | None,
    configurations: Mapping[str, Mapping[str, object]],
    zero_shot_splits: Sequence[ZeroShotSplit] | None = None,
) -> Iterator[None]:
    if directory is None:
        yield
        return

    root = directory.resolve()
    spec = get_method(method)
    original = BaseMethodMeta.registry[method]

    with _observe_loaded_series() as loaded:
        def train_curve(instance: Any, ts_train: Any) -> None:
            identity = loaded.identify(ts_train)
            series = {identity.curve: ts_train}
            unit = _Unit(
                root,
                spec,
                schema,
                identity.dataset,
                identity.curve,
                series,
                series,
                configurations[identity.dataset],
                None,
            )
            _capture_unit(
                unit,
                instance,
                lambda: original.train_valid_phase(instance, ts_train),
            )

        def train_shared(instance: Any, tsTrains: dict[str, Any]) -> None:
            identities = [loaded.identify(series) for series in tsTrains.values()]
            datasets = {identity.dataset for identity in identities}
            if len(datasets) != 1:
                raise RuntimeError("shared Model Artifact spans multiple datasets")
            dataset = datasets.pop()
            split = None
            all_series = loaded.dataset(dataset)
            testing = all_series
            if schema == "zero_shot":
                if zero_shot_splits is None:
                    raise RuntimeError("zero-shot Model Artifact requires a frozen split")
                split = split_for_dataset(zero_shot_splits, dataset)
                if tuple(tsTrains) != split.training_curves:
                    raise RuntimeError("zero-shot training members differ from frozen split")
                testing = {
                    curve: all_series[curve] for curve in split.held_out_curves
                }
            unit = _Unit(
                root,
                spec,
                schema,
                dataset,
                None,
                tsTrains,
                testing,
                configurations[dataset],
                split,
            )
            _capture_unit(
                unit,
                instance,
                lambda: original.train_valid_phase_all_in_one(instance, tsTrains),
            )

        BaseMethodMeta(
            method,
            (original,),
            {
                "__module__": original.__module__,
                "train_valid_phase": train_curve,
                "train_valid_phase_all_in_one": train_shared,
            },
        )
        try:
            yield
        finally:
            BaseMethodMeta.registry[method] = original


__all__ = ["bind_model_artifacts"]
