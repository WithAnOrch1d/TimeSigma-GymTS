"""Official OmniAnomaly through a label-free numeric Python 3.6/TF1 worker."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path

import numpy as np
from EasyTSAD.Controller import TSADController as _TSADController
from EasyTSAD.Methods import BaseMethod

from .client import WorkerClient
from .spec import SPEC

_EXECUTION = ContextVar("omnianomaly_execution", default=None)


@contextmanager
def execution_scope(directory, configuration, *, reference=False, repeat_prediction=False):
    """Bind one NAIVE unit without publishing a Method in the Agent Catalog."""
    from .binding import validate_configuration

    validate_configuration(configuration)
    instances = []
    token = _EXECUTION.set({"directory": Path(directory), "configuration": deepcopy(configuration),
                            "reference": reference, "repeat_prediction": repeat_prediction, "instances": instances})
    try:
        yield instances
    finally:
        for instance in instances:
            if instance.worker is not None:
                instance.worker.close()
        _EXECUTION.reset(token)


def validate_lengths(train, valid, test, parameters):
    width, batch = parameters["window_length"], parameters["batch_size"]
    if min(train, valid, test) < width or int((train + valid) * .2) != valid:
        raise ValueError("OmniAnomaly requires complete native windows and the canonical raw 80/20 split")
    steps = parameters["max_epoch"] * ((train - width + 1) // batch)
    if steps < parameters["valid_step_freq"]:
        raise ValueError("OmniAnomaly canonical unit cannot reach native validation with the frozen defaults")
    return SPEC.score_support(test, parameters)


class OmniAnomaly(BaseMethod):
    spec = SPEC

    def __init__(self, params):
        from .binding import validate_configuration
        from timesigma.algorithms.configuration import resolve_configuration

        configuration = resolve_configuration(SPEC.module, ("Default",))["Default"]
        configuration["Model_Params"] = deepcopy(dict(params))
        validate_configuration(configuration)
        self.params = deepcopy(dict(params))
        self.window_length = params["window_length"]
        self.configuration = configuration
        self.worker = None
        self.result = self._anomaly_score = None
        self.replay_bundle = None
        self.replay_cursor = 0
        self.context = _EXECUTION.get()
        if self.context is not None:
            if self.context["instances"]:
                raise ValueError("OmniAnomaly execution scope requires exactly one NAIVE curve")
            self.context["instances"].append(self)

    def train_valid_phase(self, tsTrain):
        if self.context is None or self.worker is not None or self.replay_bundle is not None:
            raise RuntimeError("OmniAnomaly fitting requires a fresh bounded NAIVE execution scope")
        validate_lengths(len(tsTrain.train), len(tsTrain.valid), len(tsTrain.test), self.params)
        self.worker = WorkerClient(self.context["directory"] / "worker", self.configuration,
                                   reference=self.context["reference"])
        self.training = self.worker.call({"action": "train", "train": self.worker.numeric(tsTrain.train),
                                         "valid": self.worker.numeric(tsTrain.valid)})

    def train_valid_phase_all_in_one(self, tsTrains):
        raise NotImplementedError("OmniAnomaly ALL_IN_ONE and ZERO_SHOT are not implemented")

    def test_phase(self, tsData):
        if self.replay_bundle is not None:
            return self._replay(tsData)
        if self.worker is None or self.result is not None:
            raise RuntimeError("OmniAnomaly scoring requires completed native training")
        record = self.worker.call({"action": "score", "values": self.worker.numeric(tsData.test)})
        self._anomaly_score = self.worker.score(record)
        if self.context["repeat_prediction"]:
            repeated = self.worker.call({"action": "score", "values": self.worker.numeric(tsData.test)})
            self.repeated_score = self.worker.score(repeated)
        self.result = self.worker.call({"action": "finish"})
        self.worker.close()

    def _replay(self, tsData):
        from .binding import array_binding

        candidates = [item for item in self.replay_manifest["occurrences"] if item["role"] == "test"]
        if self.replay_cursor >= len(candidates):
            raise ValueError("OmniAnomaly replay has no remaining recorded input occurrence")
        item = candidates[self.replay_cursor]
        values = np.ascontiguousarray(np.asarray(tsData.test).reshape(-1, 1), dtype=np.float32)
        if array_binding(values) != item["input"]:
            raise ValueError("OmniAnomaly replay input differs from the frozen occurrence")
        # Each reload uses a fresh process, even for an identical input occurrence.
        directory = self.replay_output / f"occurrence-{self.replay_cursor}"
        with WorkerClient(directory, self.configuration, seconds=600) as worker:
            self.reload_result = worker.call({"action": "load", "bundle": str(self.replay_bundle)})
            record = worker.call({"action": "replay", "values": worker.numeric(values), "occurrence": item["ordinal"]})
            self._anomaly_score = worker.score(record)
        self.replay_cursor += 1

    def anomaly_score(self):
        if self._anomaly_score is None:
            raise RuntimeError("OmniAnomaly has not produced an anomaly score")
        return self._anomaly_score
