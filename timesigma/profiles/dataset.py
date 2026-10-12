from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from timesigma.algorithms import EASYTSAD_SCHEMAS
from timesigma.benchmark.zero_shot import (
    ZeroShotSplit,
    load_zero_shot_membership,
    split_for_dataset,
)

from .features import CurveProfile, PROFILE_FEATURES, profile_curve
from .store import load_curve_profile_names, load_curve_profiles


@dataclass(frozen=True)
class FeatureSummary:
    defined_curves: int
    mean: float | None
    standard_deviation: float | None
    median: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "defined_curves": self.defined_curves,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "median": self.median,
        }


@dataclass(frozen=True)
class DatasetAggregate:
    curve_count: int
    total_points: int
    constant_curve_fraction: float
    features: Mapping[str, FeatureSummary]

    def as_dict(self) -> dict[str, object]:
        return {
            "curve_count": self.curve_count,
            "total_points": self.total_points,
            "constant_curve_fraction": self.constant_curve_fraction,
            "features": {
                name: summary.as_dict()
                for name, summary in self.features.items()
            },
        }


@dataclass(frozen=True)
class DatasetProfile:
    dataset: str
    source: str
    revision: str
    curve: str | None
    training_curves: tuple[str, ...]
    held_out_curves: tuple[str, ...]
    curve_profile: CurveProfile | None
    aggregate: DatasetAggregate | None
    zero_shot_split: ZeroShotSplit | None

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "source": self.source,
            "revision": self.revision,
            "input_state": "raw_train",
            "curve": self.curve,
            "training_curves": list(self.training_curves),
            "held_out_curves": list(self.held_out_curves),
            "curve_profile": (
                None if self.curve_profile is None else self.curve_profile.as_dict()
            ),
            "aggregate": None if self.aggregate is None else self.aggregate.as_dict(),
            "zero_shot_split": (
                None
                if self.zero_shot_split is None
                else self.zero_shot_split.as_dict()
            ),
        }


def build_dataset_profiles(
    *,
    dataset: str,
    dataset_dir: Path,
    schema: str,
    provenance: Mapping[str, object],
    zero_shot_membership: Path | None = None,
    profile_dir: Path | None = None,
) -> tuple[DatasetProfile, ...]:
    if schema not in EASYTSAD_SCHEMAS:
        raise ValueError(f"unsupported EasyTSAD schema: {schema}")
    source = provenance.get("source")
    revision = provenance.get("revision")
    if not isinstance(source, str) or not isinstance(revision, str):
        raise ValueError("a Dataset Profile requires source and revision provenance")

    root = dataset_dir.resolve()
    curve_names = _curve_names(
        root=root,
        dataset=dataset,
        provenance=provenance,
        profile_dir=profile_dir,
    )

    if schema == "zero_shot":
        if zero_shot_membership is None:
            raise ValueError("zero_shot Profile requires frozen membership")
        split = split_for_dataset(
            load_zero_shot_membership(zero_shot_membership), dataset
        )
        if set(split.enumerated_curves) != set(curve_names):
            raise ValueError("zero_shot Profile membership does not match the dataset")
        if not split.training_curves:
            raise ValueError("zero_shot Profile requires at least one training curve")
        profiles = _profiles_for_curves(
            root=root,
            dataset=dataset,
            provenance=provenance,
            curve_names=tuple(sorted(split.training_curves)),
            profile_dir=profile_dir,
        )
        return (
            _dataset_profile(
                dataset=dataset,
                source=source,
                revision=revision,
                curve=None,
                training_curves=split.training_curves,
                held_out_curves=split.held_out_curves,
                curve_profile=None,
                aggregate=_aggregate(profiles),
                split=split,
            ),
        )

    if zero_shot_membership is not None:
        raise ValueError("frozen membership is only valid for a zero_shot Profile")
    profiles = _profiles_for_curves(
        root=root,
        dataset=dataset,
        provenance=provenance,
        curve_names=curve_names,
        profile_dir=profile_dir,
    )
    if schema == "naive":
        return tuple(
            _dataset_profile(
                dataset=dataset,
                source=source,
                revision=revision,
                curve=curve,
                training_curves=(curve,),
                held_out_curves=(),
                curve_profile=profiles[curve],
                aggregate=None,
                split=None,
            )
            for curve in curve_names
        )
    return (
        _dataset_profile(
            dataset=dataset,
            source=source,
            revision=revision,
            curve=None,
            training_curves=curve_names,
            held_out_curves=(),
            curve_profile=None,
            aggregate=_aggregate(profiles),
            split=None,
        ),
    )


def _load_profiles(root: Path, curve_names: tuple[str, ...]) -> dict[str, CurveProfile]:
    return {
        curve: profile_curve(
            np.load(root / curve / "train.npy", allow_pickle=False)
        )
        for curve in curve_names
    }


def _curve_names(
    *,
    root: Path,
    dataset: str,
    provenance: Mapping[str, object],
    profile_dir: Path | None,
) -> tuple[str, ...]:
    if root.is_dir():
        curve_names = tuple(sorted(path.name for path in root.iterdir() if path.is_dir()))
        if not curve_names:
            raise ValueError(f"Profile dataset has no curves: {root}")
        return curve_names
    if profile_dir is None:
        raise FileNotFoundError(f"Profile dataset directory is missing: {root}")
    return load_curve_profile_names(
        dataset=dataset,
        profile_dir=profile_dir,
        provenance=provenance,
    )


def _profiles_for_curves(
    *,
    root: Path,
    dataset: str,
    provenance: Mapping[str, object],
    curve_names: tuple[str, ...],
    profile_dir: Path | None,
) -> dict[str, CurveProfile]:
    if profile_dir is None:
        return _load_profiles(root, curve_names)
    return load_curve_profiles(
        dataset=dataset,
        profile_dir=profile_dir,
        provenance=provenance,
        curve_names=curve_names,
    )


def _aggregate(
    profiles: Mapping[str, CurveProfile],
) -> DatasetAggregate:
    names = ("point_count", *PROFILE_FEATURES)
    summaries: dict[str, FeatureSummary] = {}
    for name in names:
        values = [
            value
            for profile in profiles.values()
            if (value := getattr(profile, name)) is not None
        ]
        if not values:
            summaries[name] = FeatureSummary(0, None, None, None)
            continue
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("a Dataset Profile aggregate requires finite values")
        scale = float(np.max(np.abs(array)))
        scaled = array if scale == 0.0 else array / scale
        mean = float(np.mean(scaled)) * scale
        standard_deviation = float(np.std(scaled)) * scale
        median = float(np.median(scaled)) * scale
        if not np.isfinite((mean, standard_deviation, median)).all():
            raise ValueError("a Dataset Profile aggregate is not finite")
        summaries[name] = FeatureSummary(
            defined_curves=int(array.size),
            mean=mean,
            standard_deviation=standard_deviation,
            median=median,
        )
    return DatasetAggregate(
        curve_count=len(profiles),
        total_points=sum(profile.point_count for profile in profiles.values()),
        constant_curve_fraction=sum(
            profile.is_constant for profile in profiles.values()
        )
        / len(profiles),
        features=summaries,
    )


def _dataset_profile(
    *,
    dataset: str,
    source: str,
    revision: str,
    curve: str | None,
    training_curves: tuple[str, ...],
    held_out_curves: tuple[str, ...],
    curve_profile: CurveProfile | None,
    aggregate: DatasetAggregate | None,
    split: ZeroShotSplit | None,
) -> DatasetProfile:
    return DatasetProfile(
        dataset=dataset,
        source=source,
        revision=revision,
        curve=curve,
        training_curves=training_curves,
        held_out_curves=held_out_curves,
        curve_profile=curve_profile,
        aggregate=aggregate,
        zero_shot_split=split,
    )


__all__ = [
    "DatasetAggregate",
    "DatasetProfile",
    "FeatureSummary",
    "build_dataset_profiles",
]
