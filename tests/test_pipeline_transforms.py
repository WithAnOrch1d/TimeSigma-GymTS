from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from timesigma.research import (
    apply_preprocessing_components,
    apply_score_postprocessing,
    fuse_scores_mean,
    materialize_dataset_view,
)


def test_preprocessing_components_are_label_free_series_transforms() -> None:
    train = np.asarray([1.0, 2.0, 100.0, 4.0, 5.0])
    test = np.asarray([2.0, 3.0, 4.0, 5.0, 6.0])

    transformed_train, transformed_test = apply_preprocessing_components(
        train=train,
        test=test,
        components=(
            {"component_id": "robust_scaler", "parameters": {}},
            {"component_id": "moving_average", "parameters": {"window": 3}},
        ),
    )

    assert transformed_train.shape == train.shape
    assert transformed_test.shape == test.shape
    assert np.isfinite(transformed_train).all()
    assert np.isfinite(transformed_test).all()
    assert not np.array_equal(transformed_train, train)


def test_materialize_dataset_view_writes_transformed_arrays_and_copies_metadata(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "source/TODS"
    curve = dataset_dir / "0"
    curve.mkdir(parents=True)
    np.save(curve / "train.npy", np.asarray([1.0, 2.0, 3.0]), allow_pickle=False)
    np.save(curve / "test.npy", np.asarray([4.0, 5.0, 6.0]), allow_pickle=False)
    np.save(curve / "train_label.npy", np.asarray([0, 0, 1]), allow_pickle=False)
    np.save(curve / "test_label.npy", np.asarray([0, 1, 0]), allow_pickle=False)
    (curve / "info.json").write_text('{"source":"fixture"}\n', encoding="utf-8")

    output_root = materialize_dataset_view(
        dataset="TODS",
        dataset_dir=dataset_dir,
        output_root=tmp_path / "view",
        curve_names=("0",),
        preprocessing=({"component_id": "robust_scaler", "parameters": {}},),
    )

    view_curve = output_root / "UTS/TODS/0"
    assert np.load(view_curve / "train.npy", allow_pickle=False).shape == (3,)
    assert np.load(view_curve / "test.npy", allow_pickle=False).shape == (3,)
    assert np.array_equal(
        np.load(view_curve / "test_label.npy", allow_pickle=False),
        np.asarray([0, 1, 0]),
    )
    metadata = json.loads((output_root / "dataset-view.json").read_text())
    assert metadata["dataset"] == "TODS"
    assert metadata["curve_names"] == ["0"]


def test_score_postprocessing_and_fusion_are_shape_safe() -> None:
    first = np.asarray([1.0, 2.0, 3.0, 4.0])
    second = np.asarray([2.0, 3.0, 4.0, 5.0])

    smoothed = apply_score_postprocessing(
        first,
        ({"component_id": "score_moving_average", "parameters": {"window": 3}},),
    )
    fused = fuse_scores_mean((first, second))

    assert smoothed.shape == first.shape
    assert np.allclose(fused, np.asarray([1.5, 2.5, 3.5, 4.5]))
    with pytest.raises(ValueError, match="equal score shapes"):
        fuse_scores_mean((first, np.asarray([1.0, 2.0])))
