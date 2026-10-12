from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import timesigma.profiles.dataset as profile_dataset
import timesigma.profiles.store as profile_store
from timesigma.profiles import (
    CURVE_PROFILE_DEFINITION,
    build_dataset_profiles,
    materialize_curve_profiles,
    profile_curve,
)


PROVENANCE = {"source": "https://example.test/datasets.git", "revision": "abc123"}


def test_curve_profile_has_fixed_interpretable_values() -> None:
    ramp = profile_curve(np.arange(5.0))

    assert ramp.point_count == 5
    assert ramp.minimum == 0.0
    assert ramp.q25 == 1.0
    assert ramp.median == 2.0
    assert ramp.q75 == 3.0
    assert ramp.maximum == 4.0
    assert ramp.mean == 2.0
    assert ramp.standard_deviation == pytest.approx(np.sqrt(2.0))
    assert ramp.skewness == pytest.approx(0.0)
    assert ramp.excess_kurtosis == pytest.approx(-1.3)
    assert ramp.difference_standard_deviation == 0.0
    assert ramp.lag1_autocorrelation == pytest.approx(0.4)
    assert ramp.normalized_trend_slope == pytest.approx(4.0 / np.sqrt(2.0))
    assert ramp.trend_strength == pytest.approx(1.0)
    assert ramp.roughness == pytest.approx(1.0 / np.sqrt(2.0))

    alternating = profile_curve(np.tile([-1.0, 1.0], 4))
    assert alternating.lag1_autocorrelation == pytest.approx(-7.0 / 8.0)
    assert alternating.dominant_frequency == pytest.approx(0.5)
    assert alternating.spectral_entropy == pytest.approx(0.0)


def test_curve_profile_handles_constant_and_short_series_without_nan() -> None:
    constant = profile_curve(np.full(4, 5.0))

    assert constant.is_constant is True
    assert constant.skewness == 0.0
    assert constant.excess_kurtosis == 0.0
    assert constant.difference_standard_deviation == 0.0
    assert constant.lag1_autocorrelation is None
    assert constant.normalized_trend_slope == 0.0
    assert constant.trend_strength == 0.0
    assert constant.dominant_frequency is None
    assert constant.spectral_entropy is None
    assert constant.roughness == 0.0

    singleton = profile_curve(np.array([3.0]))
    assert singleton.difference_standard_deviation is None
    assert singleton.lag1_autocorrelation is None
    assert singleton.normalized_trend_slope is None
    assert singleton.trend_strength is None
    assert singleton.roughness is None
    assert all(
        value is None or isinstance(value, (bool, int)) or np.isfinite(value)
        for value in singleton.as_dict().values()
    )


@pytest.mark.parametrize(
    "values",
    [
        np.array([]),
        np.ones((2, 2)),
        np.array([True, False]),
        np.array([1 + 2j]),
        np.array(["1"], dtype=object),
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
    ],
)
def test_curve_profile_rejects_invalid_input(values: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        profile_curve(values)


def test_profiles_are_repeatable_and_do_not_modify_the_input() -> None:
    values = np.sin(np.linspace(0.0, 8.0 * np.pi, 128))
    original = values.copy()

    first = profile_curve(values)
    second = profile_curve(values)

    assert first == second
    assert np.array_equal(values, original)


def test_materialized_profiles_are_reused_without_loading_training_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "UTS/TODS"
    _write_curve(dataset_dir, "curve-a", np.arange(6.0))
    _write_curve(dataset_dir, "curve-b", np.arange(4.0))
    profile_dir = tmp_path / "profiles/TODS/abc123"
    loaded: list[Path] = []
    original_load = np.load

    def recording_load(path: Path, **kwargs: object) -> np.ndarray:
        loaded.append(Path(path))
        return original_load(path, **kwargs)

    monkeypatch.setattr(profile_store.np, "load", recording_load)
    assert materialize_curve_profiles(
        dataset="TODS",
        dataset_dir=dataset_dir,
        profile_dir=profile_dir,
        provenance=PROVENANCE,
    ) == ("curve-a", "curve-b")
    assert [path.relative_to(dataset_dir) for path in loaded] == [
        Path("curve-a/train.npy"),
        Path("curve-b/train.npy"),
    ]

    def unexpected_load(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("materialized Profile reloaded train.npy")

    monkeypatch.setattr(profile_store.np, "load", unexpected_load)
    materialize_curve_profiles(
        dataset="TODS",
        dataset_dir=dataset_dir,
        profile_dir=profile_dir,
        provenance=PROVENANCE,
    )
    naive = build_dataset_profiles(
        dataset="TODS",
        dataset_dir=dataset_dir,
        schema="naive",
        provenance=PROVENANCE,
        profile_dir=profile_dir,
    )
    all_in_one = build_dataset_profiles(
        dataset="TODS",
        dataset_dir=dataset_dir,
        schema="all_in_one",
        provenance=PROVENANCE,
        profile_dir=profile_dir,
    )[0]

    assert tuple(profile.curve for profile in naive) == ("curve-a", "curve-b")
    assert all_in_one.aggregate is not None
    assert all_in_one.aggregate.curve_count == 2


def test_materialized_profile_bytes_are_deterministic(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "UTS/TODS"
    _write_curve(dataset_dir, "curve-b", np.arange(5.0))
    _write_curve(dataset_dir, "curve-a", np.arange(3.0))
    first = tmp_path / "profiles-first"
    second = tmp_path / "profiles-second"

    for destination in (first, second):
        materialize_curve_profiles(
            dataset="TODS",
            dataset_dir=dataset_dir,
            profile_dir=destination,
            provenance=PROVENANCE,
        )

    first_files = {
        path.relative_to(first): path.read_bytes()
        for path in sorted(first.rglob("*.json"))
    }
    second_files = {
        path.relative_to(second): path.read_bytes()
        for path in sorted(second.rglob("*.json"))
    }
    assert first_files == second_files
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile_definition"] == CURVE_PROFILE_DEFINITION
    assert "schema" not in manifest
    assert "method" not in manifest


def test_materialized_profile_identity_mismatch_fails(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "UTS/TODS"
    _write_curve(dataset_dir, "curve-a", np.arange(4.0))
    profile_dir = tmp_path / "profiles"
    materialize_curve_profiles(
        dataset="TODS",
        dataset_dir=dataset_dir,
        profile_dir=profile_dir,
        provenance=PROVENANCE,
    )

    with pytest.raises(ValueError, match="revision mismatch"):
        build_dataset_profiles(
            dataset="TODS",
            dataset_dir=dataset_dir,
            schema="naive",
            provenance={**PROVENANCE, "revision": "different"},
            profile_dir=profile_dir,
        )

    manifest_path = profile_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["profile_definition"] = "different_curve_statistics"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="profile_definition mismatch"):
        build_dataset_profiles(
            dataset="TODS",
            dataset_dir=dataset_dir,
            schema="naive",
            provenance=PROVENANCE,
            profile_dir=profile_dir,
        )


def test_materialized_profile_rejects_nonfinite_facts(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "UTS/TODS"
    _write_curve(dataset_dir, "curve-a", np.arange(4.0))
    profile_dir = tmp_path / "profiles"
    materialize_curve_profiles(
        dataset="TODS",
        dataset_dir=dataset_dir,
        profile_dir=profile_dir,
        provenance=PROVENANCE,
    )
    curve_path = profile_dir / "curves/curve-a.json"
    curve_path.write_text(
        curve_path.read_text(encoding="utf-8").replace(
            '"mean": 1.5',
            '"mean": NaN',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contains NaN"):
        build_dataset_profiles(
            dataset="TODS",
            dataset_dir=dataset_dir,
            schema="naive",
            provenance=PROVENANCE,
            profile_dir=profile_dir,
        )


def test_aiops_profiles_cannot_be_materialized(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "UTS/AIOPS"
    _write_curve(dataset_dir, "curve-a", np.arange(4.0))
    profile_dir = tmp_path / "profiles/AIOPS"

    with pytest.raises(ValueError, match="cannot be persisted"):
        materialize_curve_profiles(
            dataset="AIOPS",
            dataset_dir=dataset_dir,
            profile_dir=profile_dir,
            provenance=PROVENANCE,
        )

    assert not profile_dir.exists()


def test_naive_and_all_in_one_share_curve_facts_without_concatenation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "UTS/TODS"
    _write_curve(dataset_dir, "long", np.full(4, 10.0))
    _write_curve(dataset_dir, "short", np.zeros(2))

    loaded: list[Path] = []
    original_load = np.load

    def recording_load(path: Path, **kwargs: object) -> np.ndarray:
        loaded.append(Path(path))
        assert kwargs == {"allow_pickle": False}
        return original_load(path, **kwargs)

    monkeypatch.setattr(profile_dataset.np, "load", recording_load)
    naive = build_dataset_profiles(
        dataset="TODS",
        dataset_dir=dataset_dir,
        schema="naive",
        provenance=PROVENANCE,
    )
    all_in_one = build_dataset_profiles(
        dataset="TODS",
        dataset_dir=dataset_dir,
        schema="all_in_one",
        provenance=PROVENANCE,
    )[0]

    assert tuple(unit.curve for unit in naive) == ("long", "short")
    assert all_in_one.training_curves == ("long", "short")
    assert naive[0].curve_profile == profile_curve(np.full(4, 10.0))
    assert naive[1].curve_profile == profile_curve(np.zeros(2))
    assert all_in_one.aggregate is not None
    assert all_in_one.aggregate.features["mean"].mean == 5.0
    assert all_in_one.aggregate.features["mean"].standard_deviation == 5.0
    assert all_in_one.aggregate.features["mean"].median == 5.0
    assert all_in_one.aggregate.features["point_count"].mean == 3.0
    assert all_in_one.aggregate.features["standard_deviation"].mean == 0.0
    assert all(path.name == "train.npy" for path in loaded)
    assert all_in_one.as_dict()["input_state"] == "raw_train"
    assert "schema" not in all_in_one.as_dict()


def test_all_in_one_aggregate_is_finite_for_extreme_curve_facts(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "UTS/TODS"
    _write_curve(dataset_dir, "a-positive", np.array([1e308]))
    _write_curve(dataset_dir, "b-positive", np.array([1e308]))
    _write_curve(dataset_dir, "c-positive", np.array([1e308]))
    _write_curve(dataset_dir, "z-negative", np.array([-1e308]))

    aggregate = build_dataset_profiles(
        dataset="TODS",
        dataset_dir=dataset_dir,
        schema="all_in_one",
        provenance=PROVENANCE,
    )[0].aggregate

    assert aggregate is not None
    mean_summary = aggregate.features["mean"]
    assert mean_summary.mean == pytest.approx(5e307)
    assert mean_summary.standard_deviation == pytest.approx(np.sqrt(0.75) * 1e308)
    assert mean_summary.median == pytest.approx(1e308)
    payload = aggregate.as_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_zero_shot_profiles_only_frozen_training_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "UTS/TODS"
    _write_curve(dataset_dir, "train-curve", np.arange(8.0))
    (dataset_dir / "held-out-curve").mkdir(parents=True)
    membership = _write_membership(
        tmp_path / "membership.json",
        enumerated=("held-out-curve", "train-curve"),
        training=("train-curve",),
        held_out=("held-out-curve",),
    )
    loaded: list[Path] = []
    original_load = np.load

    def recording_load(path: Path, **kwargs: object) -> np.ndarray:
        loaded.append(Path(path))
        assert kwargs == {"allow_pickle": False}
        return original_load(path, **kwargs)

    monkeypatch.setattr(profile_dataset.np, "load", recording_load)
    profile = build_dataset_profiles(
        dataset="TODS",
        dataset_dir=dataset_dir,
        schema="zero_shot",
        provenance=PROVENANCE,
        zero_shot_membership=membership,
    )[0]

    assert profile.training_curves == ("train-curve",)
    assert profile.held_out_curves == ("held-out-curve",)
    assert profile.curve_profile is None
    assert profile.aggregate is not None
    assert profile.aggregate.curve_count == 1
    assert profile.aggregate.features["mean"].mean == pytest.approx(3.5)
    assert [path.relative_to(dataset_dir) for path in loaded] == [
        Path("train-curve/train.npy")
    ]


def test_zero_shot_materialized_view_does_not_read_held_out_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "UTS/TODS"
    _write_curve(dataset_dir, "train-curve", np.arange(8.0))
    _write_curve(dataset_dir, "held-out-curve", np.arange(5.0))
    profile_dir = tmp_path / "profiles"
    materialize_curve_profiles(
        dataset="TODS",
        dataset_dir=dataset_dir,
        profile_dir=profile_dir,
        provenance=PROVENANCE,
    )
    (profile_dir / "curves/held-out-curve.json").write_text(
        "not valid JSON",
        encoding="utf-8",
    )
    membership = _write_membership(
        tmp_path / "membership.json",
        enumerated=("held-out-curve", "train-curve"),
        training=("train-curve",),
        held_out=("held-out-curve",),
    )

    def unexpected_load(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("materialized zero-shot Profile loaded train.npy")

    monkeypatch.setattr(profile_store.np, "load", unexpected_load)
    profile = build_dataset_profiles(
        dataset="TODS",
        dataset_dir=dataset_dir,
        schema="zero_shot",
        provenance=PROVENANCE,
        zero_shot_membership=membership,
        profile_dir=profile_dir,
    )[0]

    assert profile.training_curves == ("train-curve",)
    assert profile.held_out_curves == ("held-out-curve",)
    assert profile.aggregate is not None
    assert profile.aggregate.curve_count == 1


def test_zero_shot_membership_is_validated_before_reading_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "UTS/TODS"
    _write_curve(dataset_dir, "curve-a", np.arange(4.0))
    _write_curve(dataset_dir, "curve-b", np.arange(4.0))
    membership = _write_membership(
        tmp_path / "membership.json",
        enumerated=("curve-a", "curve-c"),
        training=("curve-a",),
        held_out=("curve-c",),
    )

    def unexpected_load(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("curve values were read before membership validation")

    monkeypatch.setattr(profile_dataset.np, "load", unexpected_load)
    with pytest.raises(ValueError, match="membership does not match"):
        build_dataset_profiles(
            dataset="TODS",
            dataset_dir=dataset_dir,
            schema="zero_shot",
            provenance=PROVENANCE,
            zero_shot_membership=membership,
        )

    empty_training = _write_membership(
        tmp_path / "empty-membership.json",
        enumerated=("curve-a", "curve-b"),
        training=(),
        held_out=("curve-a", "curve-b"),
    )
    with pytest.raises(ValueError, match="at least one training curve"):
        build_dataset_profiles(
            dataset="TODS",
            dataset_dir=dataset_dir,
            schema="zero_shot",
            provenance=PROVENANCE,
            zero_shot_membership=empty_training,
        )


def _write_curve(root: Path, name: str, values: np.ndarray) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    np.save(directory / "train.npy", values)


def _write_membership(
    path: Path,
    *,
    enumerated: tuple[str, ...],
    training: tuple[str, ...],
    held_out: tuple[str, ...],
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "zero_shot",
                "dataset_type": "UTS",
                "datasets": [
                    {
                        "dataset": "TODS",
                        "proportion": 0.5,
                        "random_seed": 1,
                        "enumerated_curves": list(enumerated),
                        "training_curves": list(training),
                        "held_out_curves": list(held_out),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path
