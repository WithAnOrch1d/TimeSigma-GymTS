"""Library, fitted-row and score bindings inside the existing estimator Artifact."""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
from importlib import metadata
import platform

SOURCE_IDENTITY = "sklearn.ensemble.IsolationForest@1.9.0"
BINDING_FIELDS = (
    "method", "schema", "dataset", "curve", "configuration", "seed",
    "dataset_revision", "easytsad_revision", "source_identity", "source_binding",
    "runtime_binding", "training_curves", "test_curves", "fit_rows",
    "sample_counts", "score_support", "input_features", "row_order", "score_definition",
)


def source_binding():
    import sklearn

    dist = metadata.distribution("scikit-learn")
    if dist.version != "1.9.0" or sklearn.__version__ != "1.9.0":
        raise ValueError("IsolationForest requires pinned scikit-learn 1.9.0")
    # Bind the implementation and its fitted-tree/bagging machinery, including
    # compiled state semantics; RECORD verifies these installed dependency bytes.
    names = {"sklearn/ensemble/_iforest.py", "sklearn/ensemble/_bagging.py",
             "sklearn/tree/_classes.py", "sklearn/utils/validation.py"}
    selected = [p for p in dist.files if str(p) in names or
                (str(p).startswith("sklearn/tree/") and str(p).endswith(".so"))]
    if not names.issubset({str(p) for p in selected}):
        raise ValueError("IsolationForest installed source binding is incomplete")
    import base64

    files = {}
    for path in selected:
        digest = hashlib.sha256(path.locate().read_bytes()).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        if path.hash is None or path.hash.mode != "sha256" or path.hash.value != expected:
            raise ValueError(f"IsolationForest installed source differs from package RECORD: {path}")
        files[str(path)] = digest.hex()
    return {"source_kind": "python_package", "package": "scikit-learn", "version": dist.version,
            "implementation": "sklearn.ensemble.IsolationForest", "license": "BSD-3-Clause", "files": files}


def runtime_binding():
    return {"runtime_id": "main", "python": platform.python_version(), "machine": platform.machine(),
            **{name: metadata.version(name) for name in ("scikit-learn", "numpy", "scipy", "joblib")}}


def validate_data_configuration(configuration, operators=()):
    data = configuration["Data_Params"]
    if (dict(data) != {"preprocess": "raw", "diff_order": 0}
            or type(data.get("diff_order")) is not int or operators):
        raise ValueError("IsolationForest is frozen to raw, diff_order=0 and no preprocessing components")


def fit_rows(training):
    rows = []
    offset = 0
    for curve, series in training.items():
        count = len(series.train)
        rows.append({"curve": curve, "start": offset, "stop": offset + count, "count": count})
        offset += count
    return rows


def estimator_binding(metadata):
    return {key: deepcopy(metadata[key]) for key in BINDING_FIELDS}


def validate_metadata(spec, item, configuration):
    from timesigma.algorithms import EASYTSAD_REVISION
    from timesigma.algorithms.configuration import resolve_configuration

    validate_data_configuration(configuration)
    defaults = resolve_configuration(spec.module, (item["dataset"],))[item["dataset"]]["Model_Params"]
    spec.validate_parameters(configuration["Model_Params"], defaults)
    if item.get("source_identity") != SOURCE_IDENTITY or item.get("source_binding") != source_binding():
        raise ValueError("IsolationForest Artifact source/version does not match pinned implementation")
    if item.get("runtime_binding") != runtime_binding():
        raise ValueError("IsolationForest Artifact runtime does not match the qualified environment")
    if type(item.get("seed")) is not int or item["seed"] != configuration["Model_Params"]["random_state"]:
        raise ValueError("IsolationForest Artifact seed must match random_state=1")
    if "dataset_revision" not in item or item.get("easytsad_revision") != EASYTSAD_REVISION:
        raise ValueError("IsolationForest Artifact requires dataset and substrate provenance")
    if item.get("input_features") != 1 or item.get("row_order") != "training_curves order, then ascending within-member timestamps":
        raise ValueError("IsolationForest Artifact input representation/order is invalid")
    if item.get("score_definition") != "-estimator.score_samples(X_test)":
        raise ValueError("IsolationForest Artifact raw score definition is invalid")
    if item.get("best_epoch") is not None or item.get("selection_values") is not None:
        raise ValueError("IsolationForest has no epoch or validation selection")
    training, testing = item.get("training_curves"), item.get("test_curves")
    if not isinstance(training, list) or not training or len(set(training)) != len(training):
        raise ValueError("IsolationForest Artifact requires ordered training members")
    if not isinstance(testing, list) or not testing or len(set(testing)) != len(testing):
        raise ValueError("IsolationForest Artifact requires ordered test members")
    if item["schema"] == "naive" and (training != [item.get("curve")] or testing != training):
        raise ValueError("IsolationForest NAIVE task identity is invalid")
    if item["schema"] == "all_in_one" and testing != training:
        raise ValueError("IsolationForest ALL_IN_ONE member order is invalid")
    if item["schema"] == "zero_shot":
        split = item.get("zero_shot_split", {})
        if (set(training) & set(testing) or split.get("training_curves") != training
                or split.get("held_out_curves") != testing):
            raise ValueError("IsolationForest ZERO_SHOT source/held-out binding is invalid")
    rows = item.get("fit_rows")
    if not isinstance(rows, list) or [row.get("curve") for row in rows] != training:
        raise ValueError("IsolationForest Artifact fit rows differ from member order")
    offset = 0
    for row in rows:
        if (any(type(row.get(key)) is not int for key in ("start", "stop", "count"))
                or row["count"] < 0 or row["start"] != offset or row["stop"] != offset + row["count"]):
            raise ValueError("IsolationForest Artifact per-member fit row ranges are invalid")
        offset = row["stop"]
    counts = item["sample_counts"]
    if offset < 1 or any(counts.get(key) != offset for key in ("fit_samples", "train_points", "train_samples")):
        raise ValueError("IsolationForest Artifact fit sample count is invalid")
    support = item.get("score_support")
    if not isinstance(support, Mapping) or set(support) != set(testing):
        raise ValueError("IsolationForest Artifact support members are invalid")
    for curve, value in support.items():
        if not isinstance(value, Mapping) or type(value.get("stop")) is not int or value != spec.score_support(value["stop"], configuration["Model_Params"]):
            raise ValueError(f"IsolationForest Artifact score support is invalid: {curve}")
    if any(counts.get(key) != sum(row["count"] for row in support.values()) for key in ("test_points", "test_samples")):
        raise ValueError("IsolationForest Artifact test sample counts differ from score support")


def validate_estimator(estimator, item):
    from sklearn.ensemble import IsolationForest
    from sklearn.utils.validation import check_is_fitted

    if type(estimator) is not IsolationForest:
        raise ValueError("IsolationForest Artifact does not contain the declared estimator")
    check_is_fitted(estimator)
    if estimator.get_params() != item["configuration"]["Model_Params"]:
        raise ValueError("IsolationForest fitted estimator configuration is mismatched")
    count = item["sample_counts"]["fit_samples"]
    if (estimator.n_features_in_ != 1 or estimator._n_samples != count
            or estimator.max_samples_ != min(256, count) or len(estimator.estimators_) != 100
            or estimator.offset_ != -0.5):
        raise ValueError("IsolationForest fitted attributes do not match fit/sample/feature identity")
    if getattr(estimator, "_timesigma_binding", None) != estimator_binding(item):
        raise ValueError("IsolationForest fitted state and Artifact metadata binding differ")
