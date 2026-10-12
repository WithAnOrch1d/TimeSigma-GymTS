"""Point-wise, train-only scikit-learn IsolationForest in native EasyTSAD Schemas."""
from collections.abc import Mapping

import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from EasyTSAD.Controller import TSADController as _TSADController  # native import order
from EasyTSAD.Methods import BaseMethod

from .binding import source_binding


def point_rows(values):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "fiu":
        raise ValueError("IsolationForest requires a real numeric scalar series [N]")
    # Keep timestamp order and native sklearn float32 conversion/non-finite handling.
    # No imputation, scaling, windows, temporal features or score postprocessing.
    return values.reshape(-1, 1)


class IsolationForest(BaseMethod):
    def __init__(self, params: Mapping[str, object]):
        from timesigma.algorithms import get_method
        from timesigma.algorithms.configuration import resolve_configuration

        spec = get_method("IsolationForest")
        defaults = resolve_configuration(spec.module, ("Default",))["Default"]["Model_Params"]
        spec.validate_parameters(params, defaults)
        source_binding()  # fail before fitting on a different library implementation
        self.estimator = SklearnIsolationForest(**params)
        self._anomaly_score = None

    def train_valid_phase(self, tsTrain):
        self.estimator.fit(point_rows(tsTrain.train))

    def train_valid_phase_all_in_one(self, tsTrains):
        # Dict insertion order is supplied by the canonical native Schema loader.
        self.estimator.fit(np.concatenate([point_rows(series.train) for series in tsTrains.values()]))

    def test_phase(self, tsData):
        self._anomaly_score = -self.estimator.score_samples(point_rows(tsData.test))
        if self._anomaly_score.shape != (len(tsData.test),) or not np.isfinite(self._anomaly_score).all():
            raise ValueError("IsolationForest produced invalid point-wise scores")

    def anomaly_score(self):
        if self._anomaly_score is None:
            raise RuntimeError("IsolationForest has not scored an input")
        return self._anomaly_score
