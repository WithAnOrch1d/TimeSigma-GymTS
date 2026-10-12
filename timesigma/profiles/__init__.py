"""Deterministic, label-free train-side dataset profiles."""

from .dataset import (
    DatasetAggregate,
    DatasetProfile,
    FeatureSummary,
    build_dataset_profiles,
)
from .features import CURVE_PROFILE_DEFINITION, CurveProfile, profile_curve
from .store import load_curve_profiles, materialize_curve_profiles

__all__ = [
    "CURVE_PROFILE_DEFINITION",
    "CurveProfile",
    "DatasetAggregate",
    "DatasetProfile",
    "FeatureSummary",
    "build_dataset_profiles",
    "load_curve_profiles",
    "materialize_curve_profiles",
    "profile_curve",
]
