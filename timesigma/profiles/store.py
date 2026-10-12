from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .features import CURVE_PROFILE_DEFINITION, CurveProfile, profile_curve


def materialize_curve_profiles(
    *,
    dataset: str,
    dataset_dir: Path,
    profile_dir: Path,
    provenance: Mapping[str, object],
) -> tuple[str, ...]:
    if dataset.casefold() == "aiops":
        raise ValueError("AIOPS Profiles cannot be persisted")
    source, revision = _provenance(provenance)
    root = dataset_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Profile dataset directory is missing: {root}")
    curve_names = tuple(sorted(path.name for path in root.iterdir() if path.is_dir()))
    if not curve_names:
        raise ValueError(f"Profile dataset has no curves: {root}")

    manifest_path = profile_dir / "manifest.json"
    if manifest_path.exists():
        _validate_manifest(
            _read_json(manifest_path),
            dataset=dataset,
            source=source,
            revision=revision,
            curve_names=curve_names,
        )
        load_curve_profiles(
            dataset=dataset,
            profile_dir=profile_dir,
            provenance=provenance,
            curve_names=curve_names,
        )
        return curve_names

    curves_dir = profile_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    for curve in curve_names:
        profile = profile_curve(
            np.load(root / curve / "train.npy", allow_pickle=False)
        )
        _write_json(
            curves_dir / f"{curve}.json",
            {"curve": curve, "facts": profile.as_dict()},
        )
    _write_json(
        manifest_path,
        {
            "curve_names": list(curve_names),
            "dataset": dataset,
            "input_state": "raw_train",
            "profile_definition": CURVE_PROFILE_DEFINITION,
            "revision": revision,
            "source": source,
        },
    )
    return curve_names


def load_curve_profiles(
    *,
    dataset: str,
    profile_dir: Path,
    provenance: Mapping[str, object],
    curve_names: tuple[str, ...],
) -> dict[str, CurveProfile]:
    if dataset.casefold() == "aiops":
        raise ValueError("AIOPS Profiles cannot be loaded from persistent storage")
    source, revision = _provenance(provenance)
    manifest = _read_json(profile_dir / "manifest.json")
    stored_names = _validate_manifest(
        manifest,
        dataset=dataset,
        source=source,
        revision=revision,
    )
    unknown = set(curve_names) - set(stored_names)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"materialized Profile is missing curves: {names}")

    profiles: dict[str, CurveProfile] = {}
    for curve in curve_names:
        payload = _read_json(profile_dir / "curves" / f"{curve}.json")
        if payload.get("curve") != curve:
            raise ValueError(f"materialized Profile curve identity mismatch: {curve}")
        facts = payload.get("facts")
        if not isinstance(facts, dict):
            raise ValueError(f"materialized Profile facts are invalid: {curve}")
        try:
            profile = CurveProfile(**facts)
        except TypeError as exc:
            raise ValueError(f"materialized Profile facts are invalid: {curve}") from exc
        if (
            isinstance(profile.point_count, bool)
            or not isinstance(profile.point_count, int)
            or profile.point_count < 1
            or not isinstance(profile.is_constant, bool)
            or any(
                value is not None
                and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                )
                for name, value in profile.as_dict().items()
                if name not in {"point_count", "is_constant"}
            )
        ):
            raise ValueError(f"materialized Profile facts are invalid: {curve}")
        profiles[curve] = profile
    return profiles


def load_curve_profile_names(
    *,
    dataset: str,
    profile_dir: Path,
    provenance: Mapping[str, object],
) -> tuple[str, ...]:
    if dataset.casefold() == "aiops":
        raise ValueError("AIOPS Profiles cannot be loaded from persistent storage")
    source, revision = _provenance(provenance)
    return _validate_manifest(
        _read_json(profile_dir / "manifest.json"),
        dataset=dataset,
        source=source,
        revision=revision,
    )


def _provenance(provenance: Mapping[str, object]) -> tuple[str, str]:
    source = provenance.get("source")
    revision = provenance.get("revision")
    if not isinstance(source, str) or not isinstance(revision, str):
        raise ValueError("a materialized Profile requires source and revision provenance")
    return source, revision


def _validate_manifest(
    manifest: dict[str, object],
    *,
    dataset: str,
    source: str,
    revision: str,
    curve_names: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    expected = {
        "dataset": dataset,
        "source": source,
        "revision": revision,
        "input_state": "raw_train",
        "profile_definition": CURVE_PROFILE_DEFINITION,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"materialized Profile {name} mismatch")
    stored = manifest.get("curve_names")
    if not isinstance(stored, list) or not all(isinstance(name, str) for name in stored):
        raise ValueError("materialized Profile curve_names are invalid")
    stored_names = tuple(stored)
    if curve_names is not None and stored_names != curve_names:
        raise ValueError("materialized Profile curve set mismatch")
    return stored_names


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"materialized Profile JSON must be an object: {path}")
    return payload


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"materialized Profile JSON contains {value}")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "load_curve_profile_names",
    "load_curve_profiles",
    "materialize_curve_profiles",
]
