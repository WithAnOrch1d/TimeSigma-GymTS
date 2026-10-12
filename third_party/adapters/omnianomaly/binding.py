"""Native TF bundle, preprocessing, selected-state and stochastic replay binding."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import toml

from .source import DIRECTORY, adapter_binding, source_binding, validate_adapter_binding

CONFIGURATION_IDENTITY = "EASYTSAD_CANONICAL_ADAPTATION"
ARTIFACT_PATH = "tensorflow/bundle.json"


def array_binding(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_configuration(configuration):
    defaults = toml.load(DIRECTORY / "config.toml")
    expected = {"Data_Params": defaults["Data_Params"], "Model_Params": defaults["Model_Params"]["Default"]}
    if configuration != expected or set(configuration) != set(expected):
        raise ValueError("OmniAnomaly configuration is frozen; no HPO or preprocessing overrides")
    for section in expected:
        if set(configuration[section]) != set(expected[section]):
            raise ValueError("OmniAnomaly configuration fields differ")
        for name, value in expected[section].items():
            if type(configuration[section][name]) is not type(value):
                raise ValueError("OmniAnomaly configuration type differs: " + name)


def save_artifact(method, metadata, directory, *, identity, preprocessing):
    """Persist a native checkpoint collection, never a recipe or model pickle."""
    if method.result is None or method.result["replay"]["reference"]:
        raise ValueError("selected Artifact requires completed capture execution")
    selection = method.training["selection"]
    item = {**deepcopy(metadata), **deepcopy(identity),
            "configuration_identity": CONFIGURATION_IDENTITY,
            "source_binding": source_binding(), "adapter_binding": adapter_binding(),
            "runtime_binding": method.result["replay"]["runtime"],
            "preprocessing_identity": deepcopy(preprocessing),
            "native_best_step": selection["best_step"],
            "optimizer_steps": method.training["optimizer_steps"],
            "selected_model": selection["selected_model"],
            "native_validation": method.training["validations"],
            "score_support": {metadata["curve"]: method.spec.score_support(
                metadata["sample_counts"]["test_points"], metadata["configuration"]["Model_Params"])},
            "score_definition": "negative native last-point joint Gaussian log density",
            "replay_payload": "tensorflow/replay.json"}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / "tensorflow"
    shutil.copytree(method.result["directory"], destination)
    files = {str(p.relative_to(destination)): file_digest(p) for p in sorted(destination.rglob("*")) if p.is_file()}
    write(destination / "bundle.json", {"binding": item, "files": files})
    item["bundle_sha256"] = file_digest(destination / "bundle.json")
    write(directory / "metadata.json", item)
    return directory / "metadata.json"


def validate_metadata(spec, item, configuration):
    from timesigma.algorithms import EASYTSAD_REVISION
    from . import validate_lengths

    validate_configuration(configuration)
    if item.get("schema") != "naive" or item.get("configuration_identity") != CONFIGURATION_IDENTITY:
        raise ValueError("OmniAnomaly only implements canonical NAIVE adaptation")
    if item.get("source_binding") != source_binding():
        raise ValueError("OmniAnomaly Artifact source/adapter differs")
    validate_adapter_binding(item.get("adapter_binding"))
    if item.get("easytsad_revision") != EASYTSAD_REVISION or type(item.get("seed")) is not int or item["seed"] != 1:
        raise ValueError("OmniAnomaly Artifact evaluator/seed differs")
    if not isinstance(item.get("dataset_revision"), str) or not item["dataset_revision"]:
        raise ValueError("OmniAnomaly Artifact requires dataset revision")
    if item.get("training_curves") != [item.get("curve")] or item.get("test_curves") != item["training_curves"]:
        raise ValueError("OmniAnomaly Artifact must bind one complete NAIVE curve")
    if item.get("selection_status") != "available" or item.get("score_source") != "selected_artifact":
        raise ValueError("OmniAnomaly Artifact requires actual native selection")
    counts = item["sample_counts"]
    if any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError("invalid OmniAnomaly sample counts")
    support = validate_lengths(counts["train_points"], counts["valid_points"], counts["test_points"], configuration["Model_Params"])
    if item.get("score_support") != {item["curve"]: support}:
        raise ValueError("OmniAnomaly native suffix support differs")
    for phase in ("train", "valid", "test"):
        if counts[phase + "_samples"] != counts[phase + "_points"] - 99:
            raise ValueError("OmniAnomaly sample count differs from native windows")
    if counts["fit_samples"] != counts["train_samples"]:
        raise ValueError("OmniAnomaly fit sample count differs")
    steps_per_epoch = counts["train_samples"] // 50
    total = 10 * steps_per_epoch
    rows = item.get("native_validation")
    if not isinstance(rows, list) or [row.get("step") for row in rows] != list(range(100, total + 1, 100)):
        raise ValueError("OmniAnomaly native validation cadence differs")
    previous = None
    for row in rows:
        value = row.get("loss")
        if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
                or type(row.get("saved")) is not bool or row["epoch"] != math.ceil(row["step"] / steps_per_epoch)
                or row["saved"] != (previous is None or value < previous)):
            raise ValueError("OmniAnomaly validation selection/tie rule differs")
        previous = value if previous is None else min(previous, value)
    best = min(rows, key=lambda row: row["loss"])
    if (item.get("optimizer_steps") != total or item.get("native_best_step") != best["step"]
            or item.get("best_epoch") != best["epoch"] or item.get("selection_values") != [best["loss"]]):
        raise ValueError("OmniAnomaly selected native state identity differs")
    preprocessing = item.get("preprocessing_identity", {})
    if (preprocessing.get("configuration") != configuration["Data_Params"]
            or preprocessing.get("valid_proportion") != .2 or preprocessing.get("train_proportion") != 1.
            or set(preprocessing.get("processed", {})) != {"train", "valid", "test"}
            or set(preprocessing.get("raw", {})) != {"train", "valid", "test"}
            or not preprocessing.get("scaler")):
        raise ValueError("OmniAnomaly canonical preprocessing/scaler binding missing")
    for phase in ("train", "valid", "test"):
        if preprocessing["processed"][phase]["shape"] != [counts[phase + "_points"]]:
            raise ValueError("OmniAnomaly preprocessing partition length differs")


def validate_bundle(path, metadata):
    path = Path(path)
    if path.name != "bundle.json" or file_digest(path) != metadata.get("bundle_sha256"):
        raise ValueError("OmniAnomaly TF bundle manifest digest differs")
    manifest = json.loads(path.read_text())
    expected = {k: v for k, v in metadata.items() if k != "bundle_sha256"}
    if manifest.get("binding") != expected:
        raise ValueError("OmniAnomaly TF bundle and metadata binding differ")
    required = {"checkpoint/variables.dat.data-00000-of-00001", "checkpoint/variables.dat.index",
                "checkpoint/variables.dat.meta", "checkpoint/latest", "noise.npz", "training.json", "replay.json"}
    if set(manifest.get("files", {})) != required:
        raise ValueError("OmniAnomaly native checkpoint/replay file set incomplete")
    root = path.parent.resolve()
    for name, sha in manifest["files"].items():
        source = root / name
        if not source.is_file() or source.is_symlink() or not source.resolve().is_relative_to(root) or file_digest(source) != sha:
            raise ValueError("OmniAnomaly checkpoint/replay file binding differs: " + name)
    replay = json.loads((root / "replay.json").read_text())
    training = json.loads((root / "training.json").read_text())
    if (replay["configuration"] != metadata["configuration"] or replay["runtime"] != metadata["runtime_binding"]
            or replay["reference"] or training["selection"]["selected_model"] != metadata["selected_model"]
            or training["selection"]["restored_model"] != metadata["selected_model"]
            or training["validations"] != metadata["native_validation"]):
        raise ValueError("OmniAnomaly saved native facts differ from metadata")
    dependencies = replay["random_dependencies"]
    if len(dependencies) != 1 or dependencies[0]["type"] != "TruncatedNormal":
        raise ValueError("OmniAnomaly unknown score RNG dependency")
    batches = replay["batches"]
    offset = 0
    with np.load(root / "noise.npz", allow_pickle=False) as noise:
        if set(noise.files) != {f"batch_{i}" for i in range(len(batches))}:
            raise ValueError("OmniAnomaly RNG payload incomplete")
        for ordinal, occurrence in enumerate(replay["occurrences"]):
            count = occurrence["input"]["shape"][0] - 99
            batch_count = math.ceil(count / 50)
            if (occurrence["ordinal"] != ordinal or occurrence["first_batch"] != offset
                    or occurrence["batch_count"] != batch_count or occurrence["score"]["shape"] != [count]
                    or occurrence["input"]["dtype"] != "float32" or occurrence["input"]["shape"][1:] != [1]
                    or occurrence["role"] != ("train_prefix" if ordinal == 0 else "test")):
                raise ValueError("OmniAnomaly replay occurrence order/support differs")
            for index in range(batch_count):
                batch = offset + index
                record, draw = batches[batch], noise[f"batch_{batch}"]
                size = min(50, count - index * 50)
                if (record["batch"] != batch or record["occurrence"] != ordinal
                        or record["input"]["shape"] != [size, 100, 1] or record["input"]["dtype"] != "float32"
                        or record["noise_tensor"] != dependencies[0]["name"] + ":0"
                        or array_binding(draw) != record["noise"] or draw.shape != (100, 1, 3)
                        or draw.dtype != np.float32 or not np.isfinite(draw).all()):
                    raise ValueError("OmniAnomaly replay batch/noise binding differs")
            offset += batch_count
    if offset != len(batches) or len(replay["occurrences"]) < 2:
        raise ValueError("OmniAnomaly incomplete replay occurrences")
    test_length = metadata["sample_counts"]["test_points"]
    if any(o["input"]["shape"] != [test_length, 1] for o in replay["occurrences"][1:]):
        raise ValueError("OmniAnomaly replay test input support differs")
    return replay


def load_checkpoint_bundle(path, metadata, method):
    replay = validate_bundle(path, metadata)
    if method.worker is not None or method.result is not None:
        raise ValueError("OmniAnomaly reload requires a fresh Method")
    method.replay_bundle = Path(path).resolve().parent
    method.replay_manifest = replay
    if not hasattr(method, "replay_output"):
        raise ValueError("OmniAnomaly reload caller must supply a new replay_output directory")
