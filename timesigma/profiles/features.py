from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


PROFILE_FEATURES = (
    "minimum",
    "q25",
    "median",
    "q75",
    "maximum",
    "mean",
    "standard_deviation",
    "skewness",
    "excess_kurtosis",
    "difference_standard_deviation",
    "lag1_autocorrelation",
    "normalized_trend_slope",
    "trend_strength",
    "dominant_frequency",
    "spectral_entropy",
    "roughness",
)

CURVE_PROFILE_DEFINITION = "raw_train_curve_statistics"


@dataclass(frozen=True)
class CurveProfile:
    """Raw train-side facts; frequency is in cycles per sample.

    Dispersion uses population moments. Optional values are ``None`` only when
    the statistic is mathematically undefined for a constant or short series.
    """

    point_count: int
    is_constant: bool
    minimum: float
    q25: float
    median: float
    q75: float
    maximum: float
    mean: float
    standard_deviation: float
    skewness: float
    excess_kurtosis: float
    difference_standard_deviation: float | None
    lag1_autocorrelation: float | None
    normalized_trend_slope: float | None
    trend_strength: float | None
    dominant_frequency: float | None
    spectral_entropy: float | None
    roughness: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "point_count": self.point_count,
            "is_constant": self.is_constant,
            **{name: getattr(self, name) for name in PROFILE_FEATURES},
        }


def profile_curve(values: Sequence[float] | np.ndarray) -> CurveProfile:
    source = np.asarray(values)
    if source.ndim != 1 or source.size == 0:
        raise ValueError("a Profile requires a non-empty one-dimensional series")
    if source.dtype.kind not in "iuf" or source.dtype.kind == "b":
        raise TypeError("a Profile requires real numeric values")

    series = source.astype(np.float64, copy=False)
    if not np.isfinite(series).all():
        raise ValueError("a Profile series must contain only finite values")

    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            mean = float(np.mean(series))
            centered = series - mean
            standard_deviation = float(np.sqrt(np.mean(centered * centered)))
            is_constant = standard_deviation == 0.0
            q25, median, q75 = (
                float(value)
                for value in np.quantile(
                    series,
                    (0.25, 0.5, 0.75),
                    method="linear",
                )
            )
            skewness, excess_kurtosis = _shape(
                centered, standard_deviation, is_constant
            )
            differences = np.diff(series)
            difference_standard_deviation = (
                float(np.std(differences)) if differences.size else None
            )
            lag1_autocorrelation = _lag1(centered, is_constant)
            normalized_trend_slope, trend_strength = _trend(
                centered, standard_deviation, is_constant
            )
            dominant_frequency, spectral_entropy = _spectrum(
                centered, is_constant
            )
            roughness = _roughness(
                differences, standard_deviation, is_constant
            )
    except FloatingPointError as exc:
        raise ValueError("a Profile calculation produced a non-finite value") from exc

    profile = CurveProfile(
        point_count=int(series.size),
        is_constant=is_constant,
        minimum=float(np.min(series)),
        q25=q25,
        median=median,
        q75=q75,
        maximum=float(np.max(series)),
        mean=mean,
        standard_deviation=standard_deviation,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        difference_standard_deviation=difference_standard_deviation,
        lag1_autocorrelation=lag1_autocorrelation,
        normalized_trend_slope=normalized_trend_slope,
        trend_strength=trend_strength,
        dominant_frequency=dominant_frequency,
        spectral_entropy=spectral_entropy,
        roughness=roughness,
    )
    if any(
        value is not None and not np.isfinite(value)
        for name, value in profile.as_dict().items()
        if name not in {"point_count", "is_constant"}
    ):
        raise ValueError("a Profile calculation produced a non-finite value")
    return profile


def _shape(
    centered: np.ndarray,
    standard_deviation: float,
    is_constant: bool,
) -> tuple[float, float]:
    if is_constant:
        return 0.0, 0.0
    normalized = centered / standard_deviation
    return (
        float(np.mean(normalized**3)),
        float(np.mean(normalized**4) - 3.0),
    )


def _lag1(centered: np.ndarray, is_constant: bool) -> float | None:
    if centered.size < 2 or is_constant:
        return None
    return float(
        np.sum(centered[:-1] * centered[1:]) / np.sum(centered * centered)
    )


def _trend(
    centered: np.ndarray,
    standard_deviation: float,
    is_constant: bool,
) -> tuple[float | None, float | None]:
    if centered.size < 2:
        return None, None
    if is_constant:
        return 0.0, 0.0
    time = np.linspace(-1.0, 1.0, centered.size)
    slope = float(np.sum(time * centered) / np.sum(time * time))
    normalized_slope = 2.0 * slope / standard_deviation
    strength = float(
        np.sum((slope * time) ** 2) / np.sum(centered * centered)
    )
    return float(normalized_slope), float(np.clip(strength, 0.0, 1.0))


def _spectrum(
    centered: np.ndarray,
    is_constant: bool,
) -> tuple[float | None, float | None]:
    if centered.size < 4 or is_constant:
        return None, None
    spectrum = np.fft.rfft(centered)[1:]
    power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
    total_power = float(np.sum(power))
    if total_power == 0.0:
        return None, None
    probabilities = power / total_power
    dominant_frequency = (int(np.argmax(power)) + 1) / centered.size
    if probabilities.size == 1:
        entropy = 0.0
    else:
        positive = probabilities[probabilities > 0.0]
        entropy = float(
            -np.sum(positive * np.log(positive)) / np.log(probabilities.size)
        )
    return float(dominant_frequency), entropy


def _roughness(
    differences: np.ndarray,
    standard_deviation: float,
    is_constant: bool,
) -> float | None:
    if not differences.size:
        return None
    if is_constant:
        return 0.0
    return float(np.mean(np.abs(differences)) / standard_deviation)


__all__ = [
    "CURVE_PROFILE_DEFINITION",
    "CurveProfile",
    "PROFILE_FEATURES",
    "profile_curve",
]
