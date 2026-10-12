"""One bounded EasyTSAD NAIVE unit, before Catalog publication."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from EasyTSAD.DataFactory import TSData

from timesigma.algorithms import EASYTSAD_REVISION
from timesigma.algorithms.configuration import freeze_configuration, resolve_configuration
from timesigma.benchmark.artifacts import _Unit
from timesigma.benchmark.manifest import build_run_manifest, write_run_manifest
from timesigma.benchmark.runner import run_easytsad_scores

from . import execution_scope
from .binding import ARTIFACT_PATH, array_binding, file_digest, save_artifact, write
from .spec import SPEC


@contextmanager
def observe_preprocessing():
    """Observe the actual native scaler; do not replace or refit preprocessing."""
    original_norm, original_fit = TSData.min_max_norm, MinMaxScaler.fit
    records, scalers, series = [], [], []

    def fit(self, *args, **kwargs):
        result = original_fit(self, *args, **kwargs)
        scalers.append({name: getattr(self, name).tolist() for name in
                       ("data_min_", "data_max_", "data_range_", "scale_", "min_")})
        return result

    def normalize(self, *args, **kwargs):
        if records:
            raise ValueError("OmniAnomaly requires exactly one NAIVE preprocessing unit")
        raw = {phase: array_binding(getattr(self, phase)) for phase in ("train", "valid", "test")}
        result = original_norm(self, *args, **kwargs)
        if len(scalers) != 1:
            raise ValueError("OmniAnomaly requires one native train-fitted scaler")
        records.append({"raw": raw, "processed": {phase: array_binding(getattr(self, phase))
                                                  for phase in ("train", "valid", "test")},
                        "scaler": scalers[0], "sklearn": sklearn.__version__,
                        "configuration": {"preprocess": "min-max", "diff_order": 0},
                        "train_proportion": 1., "valid_proportion": .2,
                        "valid_test_clip": [-2, 3]})
        series.append(self)
        return result

    TSData.min_max_norm, MinMaxScaler.fit = normalize, fit
    try:
        yield records, series
    finally:
        TSData.min_max_norm, MinMaxScaler.fit = original_norm, original_fit


def run_naive(*, task, dataset_root, output_dir, reference=False, repeat_prediction=False):
    if task["schema"] != "naive" or task["dataset"].casefold() == "aiops":
        raise ValueError("OmniAnomaly onboarding only authorizes a non-AIOPS canonical NAIVE task")
    dataset, curve = task["dataset"], task["unit"]["curve"]
    if task["unit"]["training_curves"] != [curve] or task["unit"]["held_out_curves"]:
        raise ValueError("OmniAnomaly NAIVE unit membership differs")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    configuration = resolve_configuration(SPEC.module, (dataset,))[dataset]
    frozen = freeze_configuration(directory / "configuration.toml", {dataset: configuration})
    manifest = build_run_manifest(method_name=SPEC.method_name, module=SPEC.module, schema="naive",
                                  datasets=[dataset], seed=1, dataset_root=Path(dataset_root),
                                  runtime_python=Path(sys.executable), model_artifacts=not reference,
                                  configuration_name="default", configuration_path=frozen)
    manifest.update(label_free_execution=True, evaluation_deferred=True, task=task,
                    qualification_only=True, active_admission=False)
    write_run_manifest(directory / "run.json", manifest)
    with observe_preprocessing() as (preprocessing, series), execution_scope(
            directory, configuration, reference=reference, repeat_prediction=repeat_prediction) as instances:
        run_easytsad_scores(methods={SPEC.method_name: SPEC.module}, schema="naive", datasets=[dataset],
                           dataset_root=Path(dataset_root), output_dir=directory,
                           curve_names=[curve], configuration_paths={SPEC.method_name: frozen})
    if len(instances) != 1 or len(series) != 1:
        raise ValueError("OmniAnomaly native lifecycle did not execute exactly one canonical curve")
    method = instances[0]
    selection = method.training["selection"]
    metadata_path = None
    if not reference:
        unit = _Unit(directory / "Results/ModelArtifacts", SPEC, "naive", dataset, curve,
                     {curve: series[0]}, {curve: series[0]}, configuration, None)
        metadata = unit.metadata(method, artifact_path=ARTIFACT_PATH,
                                 selection_values=[selection["best_loss"]], best_epoch=selection["best_epoch"])
        metadata_path = save_artifact(method, metadata, unit.directory,
                                     identity={"seed": 1, "dataset_revision": manifest["dataset"]["revision"],
                                               "easytsad_revision": EASYTSAD_REVISION,
                                               "source_identity": "official-omnianomaly-native-selected"},
                                     preprocessing=preprocessing[0])
    score_path = directory / "Results/Scores/OmniAnomaly/naive" / dataset / (curve + ".npy")
    score = np.load(score_path, allow_pickle=False)
    if not np.array_equal(score, method.anomaly_score()):
        raise ValueError("EasyTSAD saved scores differ from the native worker output")
    identity = {"task": task, "configuration": configuration, "seed": 1,
                "score": {"path": str(score_path.relative_to(directory)), "sha256": file_digest(score_path),
                          **array_binding(score)},
                "metadata": None if metadata_path is None else str(metadata_path.relative_to(directory)),
                "native_training": method.training, "preprocessing": preprocessing[0],
                "worker_replay": method.result["replay"], "runtime": method.result["replay"]["runtime"],
                "reference": reference, "label_free": True}
    write(directory / "identity.json", identity)
    manifest["status"] = "complete"
    write_run_manifest(directory / "run.json", manifest)
    return identity
