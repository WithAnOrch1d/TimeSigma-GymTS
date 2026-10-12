from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from timesigma.algorithms.imports import import_method
from typing import Any

import numpy as np

# EasyTSAD's fixed revision requires Controller to initialize before Methods.
from EasyTSAD.Controller import TSADController as _TSADController
from EasyTSAD.Methods import BaseMethod, BaseMethodMeta

from timesigma.algorithms.catalog import get_method


@dataclass(frozen=True)
class _TrainingSeries:
    train: np.ndarray
    valid: np.ndarray

    @classmethod
    def from_series(cls, series: Any) -> "_TrainingSeries":
        return cls(
            train=np.array(series.train, copy=True),
            valid=np.array(series.valid, copy=True),
        )


@dataclass(frozen=True)
class _TestingSeries:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray

    @classmethod
    def from_series(cls, series: Any) -> "_TestingSeries":
        return cls(
            train=np.array(series.train, copy=True),
            valid=np.array(series.valid, copy=True),
            test=np.array(series.test, copy=True),
        )


class TimeSigma(BaseMethod):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__()
        required = {"schema", "component", "component_hparams"}
        missing = required.difference(params)
        if missing:
            raise ValueError(f"missing TimeSigma parameters: {', '.join(sorted(missing))}")

        self._schema = str(params["schema"])
        self._component_name = str(params["component"])
        component_hparams = params["component_hparams"]
        if not isinstance(component_hparams, Mapping):
            raise TypeError("component_hparams must be a mapping")

        try:
            spec = get_method(self._component_name)
        except KeyError as exc:
            raise ValueError(
                f"unknown TimeSigma component: {self._component_name}"
            ) from exc
        capability = spec.capability(self._schema)
        if not capability.supported:
            raise ValueError(
                f"{self._component_name} does not support EasyTSAD schema "
                f"{self._schema}: {capability.unavailable_reason}"
            )

        module = import_method(spec.module)
        component_class = BaseMethodMeta.registry.get(spec.method_name)
        if (
            component_class is None
            or getattr(module, spec.method_name, None) is not component_class
        ):
            raise RuntimeError(
                f"EasyTSAD component {spec.method_name} was not registered by {spec.module}"
            )
        self._component = component_class(dict(component_hparams))
        self._training_phase_complete = False
        self._score: np.ndarray | None = None

    def param_statistic(self, save_file: str) -> None:
        self._component.param_statistic(save_file)

    def train_valid_phase(self, ts_train: Any) -> None:
        if self._schema != "naive":
            raise RuntimeError(
                f"schema {self._schema} requires train_valid_phase_all_in_one"
            )
        self._component.train_valid_phase(_TrainingSeries.from_series(ts_train))
        self._training_phase_complete = True

    def train_valid_phase_all_in_one(self, ts_trains: dict[str, Any]) -> None:
        if self._schema not in {"all_in_one", "zero_shot"}:
            raise RuntimeError("schema naive requires train_valid_phase")
        self._component.train_valid_phase_all_in_one(
            {
                name: _TrainingSeries.from_series(series)
                for name, series in ts_trains.items()
            }
        )
        self._training_phase_complete = True

    def test_phase(self, ts_data: Any) -> None:
        if not self._training_phase_complete:
            raise RuntimeError("TimeSigma training phase must complete before testing")

        self._score = None
        label_free_data = _TestingSeries.from_series(ts_data)
        self._component.test_phase(label_free_data)
        score = np.asarray(self._component.anomaly_score(), dtype=float)
        if score.ndim != 1 or score.size == 0:
            raise ValueError("component score must be a non-empty one-dimensional array")
        if not np.isfinite(score).all():
            raise ValueError("component score must contain only finite values")
        if score.size > len(label_free_data.test):
            raise ValueError("component score cannot be longer than the test series")
        self._score = score.copy()

    def anomaly_score(self) -> np.ndarray:
        if self._score is None:
            raise RuntimeError("test_phase must produce a score before anomaly_score")
        return self._score


del _TSADController
