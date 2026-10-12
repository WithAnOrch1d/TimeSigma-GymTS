from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import torch

if TYPE_CHECKING:
    from EasyTSAD.Methods.AE.AE import AE
    from EasyTSAD.Methods.AR.AR import AR
    from EasyTSAD.Methods.LSTMADalpha.LSTMADalpha import LSTMADalpha
    from EasyTSAD.Methods.LSTMADbeta.LSTMADbeta import LSTMADbeta
    from EasyTSAD.Methods.FITS import FITS
    from EasyTSAD.Methods.DCdetector import DCdetector
    from EasyTSAD.Methods.FCVAE import FCVAE
    from EasyTSAD.Methods.AnomalyTransformer.AnomalyTransformer import (
        AnomalyTransformer,
    )


@dataclass(frozen=True)
class SavedCheckpoint:
    path: Path
    method: str
    values: tuple[float, ...]
    best_epoch: int


class _BestState:
    def __init__(self) -> None:
        self.values: tuple[float, ...] | None = None
        self.state: dict[str, torch.Tensor] | None = None
        self.epoch: int | None = None

    def retain(
        self,
        values: tuple[float, ...],
        model: torch.nn.Module,
        epoch: int,
    ) -> None:
        self.values = values
        self.state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        self.epoch = epoch


class _ScalarStopper:
    def __init__(self, stopper: Any, best: _BestState) -> None:
        self._stopper = stopper
        self._best = best
        self._epoch = 0

    def __call__(self, loss: float, model: torch.nn.Module) -> None:
        self._epoch += 1
        values = (float(loss),)
        _require_finite(values)
        previous = self._stopper.best_score
        self._stopper(loss, model)
        if previous != self._stopper.best_score:
            self._best.retain(values, model, self._epoch)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stopper, name)


_OBSERVERS = {
    ("EasyTSAD.Methods.FCVAE.FCVAE", "FCVAE"): "scalar",
    ("EasyTSAD.Methods.DCdetector.DCdetector", "DCdetector"): "native_checkpoint",
    ("EasyTSAD.Methods.FITS.FITS", "FITS"): "scalar",
    ("EasyTSAD.Methods.LSTMADbeta.LSTMADbeta", "LSTMADbeta"): "scalar",
    ("EasyTSAD.Methods.LSTMADalpha.LSTMADalpha", "LSTMADalpha"): "scalar",
    ("EasyTSAD.Methods.AR.AR", "AR"): "scalar",
    ("EasyTSAD.Methods.AE.AE", "AE"): "scalar",
    (
        "EasyTSAD.Methods.AnomalyTransformer.AnomalyTransformer",
        "AnomalyTransformer",
    ): "pair",
}


def _require_finite(values: tuple[float, ...]) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("model-selection objective is non-finite")


def _method_contract(method: Any) -> tuple[str, str]:
    for method_type in type(method).__mro__:
        observer = _OBSERVERS.get((method_type.__module__, method_type.__name__))
        if observer is not None:
            return method_type.__name__, observer
    raise TypeError(f"checkpoint adaptation is unavailable for {type(method).__name__}")


@contextmanager
def retain_best(
    method: AR | AE | AnomalyTransformer | LSTMADalpha | LSTMADbeta | FITS | DCdetector | FCVAE,
    destination: Path,
    on_saved: Callable[[SavedCheckpoint], None] | None = None,
) -> Iterator[None]:
    """Persist and restore the best state observed by one EasyTSAD training call."""
    destination = Path(destination)
    best = _BestState()
    method_name, observer = _method_contract(method)

    if observer == "native_checkpoint":
        original_vali = method.vali
        epoch = 0

        def observe_native_validation(loader):
            nonlocal epoch
            epoch += 1
            value = original_vali(loader)
            values = (float(value),)
            _require_finite(values)
            # Native DCdetector uses delta=0 and replaces the checkpoint on ties.
            if best.values is None or values[0] <= best.values[0]:
                best.retain(values, method.model, epoch)
            return value

        method.vali = observe_native_validation

        def restore_observer() -> None:
            method.vali = original_vali

    elif observer == "scalar":
        stopper = method.early_stopping
        method.early_stopping = _ScalarStopper(stopper, best)

        def restore_observer() -> None:
            method.early_stopping = stopper

    else:
        instance_vali = method.__dict__.get("vali")
        vali = method.vali
        stopper = method.early_stopping
        epoch = 0

        def observed_vali(*args: Any, **kwargs: Any) -> tuple[float, float]:
            nonlocal epoch
            epoch += 1
            loss1, loss2 = vali(*args, **kwargs)
            values = (float(loss1), float(loss2))
            _require_finite(values)
            if best.values is None or all(
                loss <= previous - stopper.delta
                for loss, previous in zip(values, best.values, strict=True)
            ):
                best.retain(values, method.model, epoch)
            return loss1, loss2

        method.vali = observed_vali

        def restore_observer() -> None:
            if instance_vali is None:
                del method.vali
            else:
                method.vali = instance_vali

    try:
        yield
    finally:
        restore_observer()

    if best.state is None or best.values is None or best.epoch is None:
        raise RuntimeError(f"{method_name} training produced no selected state")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best.state, destination)
    load_checkpoint(method, destination)
    if on_saved is not None:
        on_saved(
            SavedCheckpoint(
                path=destination,
                method=method_name,
                values=best.values,
                best_epoch=best.epoch,
            )
        )


def load_checkpoint(
    method: AR | AE | AnomalyTransformer | LSTMADalpha | LSTMADbeta | FITS | DCdetector | FCVAE,
    source: Path,
) -> None:
    state = torch.load(Path(source), map_location="cpu", weights_only=True)
    if any(base.__module__ == "EasyTSAD.Methods.FCVAE.FCVAE" and base.__name__ == "FCVAE" for base in type(method).__mro__):
        from .fcvae import REPLAY_KEY

        method._fcvae_replay = state.pop(REPLAY_KEY, None)
        method._fcvae_replay_positions = {}
    method.model.load_state_dict(state)
    if any(base.__module__ == "EasyTSAD.Methods.DCdetector.DCdetector" and base.__name__ == "DCdetector" for base in type(method).__mro__):
        # Its native test_phase reloads a fixed filename. Bind that read to the
        # explicitly loaded Artifact, without changing the native score path.
        frozen_source = Path(source).resolve()
        native_test = type(method).test_phase.__get__(method, type(method))
        original_load = torch.load

        def test_with_artifact(series):
            expected = Path(method.save_path) / "DCdetector_checkpoint.pth"

            def bound_load(path, *args, **kwargs):
                if Path(path).resolve() == expected.resolve():
                    return original_load(frozen_source, map_location="cpu", weights_only=True)
                return original_load(path, *args, **kwargs)

            with torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled, benchmark=False,
                deterministic=True, allow_tf32=torch.backends.cudnn.allow_tf32,
            ), patch.object(torch, "load", bound_load):
                return native_test(series)

        method.test_phase = test_with_artifact
