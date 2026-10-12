"""EasyTSAD-first time-series anomaly detection experiments."""

from typing import TYPE_CHECKING

from .algorithms import (
    EASYTSAD_SCHEMAS,
    EASYTSAD_REVISION,
    METHOD_CATALOG,
    AlgorithmSpec,
    SchemaCapability,
    get_method,
)

if TYPE_CHECKING:
    from .method import TimeSigma

__version__ = "0.1.0"

__all__ = [
    "EASYTSAD_SCHEMAS",
    "EASYTSAD_REVISION",
    "METHOD_CATALOG",
    "AlgorithmSpec",
    "SchemaCapability",
    "TimeSigma",
    "get_method",
]


def __getattr__(name: str) -> object:
    if name == "TimeSigma":
        from .method import TimeSigma

        return TimeSigma
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
