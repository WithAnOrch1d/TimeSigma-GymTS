"""Label-free FCVAE inputs and replay of its unchanged stochastic inference."""
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .fcvae_source import import_fcvae_source

REPLAY_KEY = "_timesigma_inference_rng"


def _unlabeled_input(series, phase):
    values = getattr(series, phase)
    return SimpleNamespace(**{phase: values, phase + "_label": np.zeros_like(values)})


def inference_key(series) -> str:
    values = np.ascontiguousarray(series.test)
    digest = sha256(str((values.dtype.str, values.shape)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def inference_rng(device):
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_inference_rng(state, device):
    if ("cuda" in state) != (device.type == "cuda"):
        raise ValueError("FCVAE stochastic replay requires the original CPU/CUDA execution mode")
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def save_replay(method, path: Path):
    state = dict(method.model.state_dict())
    state[REPLAY_KEY] = method._fcvae_inference_rng
    torch.save(state, path)


def import_fcvae():
    package = import_fcvae_source()
    module = import_module("EasyTSAD.Methods.FCVAE.FCVAE")
    method = package.FCVAE
    if getattr(method, "_timesigma_label_free", False):
        return package
    curve_dataset, shared_dataset = module.OneByOneDataset, module.AllInOneDataset
    original_init, original_test = method.__init__, method.test_phase

    def curve_input(tsData, phase, window_size):
        return curve_dataset(_unlabeled_input(tsData, phase), phase, window_size)

    def shared_input(datas, phase, window_size):
        return shared_dataset({name: _unlabeled_input(series, phase) for name, series in datas.items()}, phase, window_size)

    def initialize(self, params):
        original_init(self, params)
        self.window = self.hp.window
        self._fcvae_inference_rng = {}

    def test(self, series):
        key = inference_key(series)
        replay = getattr(self, "_fcvae_replay", None)
        if replay is not None:
            if key not in replay:
                raise ValueError("FCVAE Artifact has no recorded inference RNG for these transformed test values")
            position = self._fcvae_replay_positions.get(key, 0)
            if position >= len(replay[key]):
                raise ValueError("FCVAE replay exhausted the recorded occurrences; reload the Artifact to replay again")
            restore_inference_rng(replay[key][position], self.device)
            self._fcvae_replay_positions[key] = position + 1
        self._fcvae_inference_rng.setdefault(key, []).append(inference_rng(self.device))
        # Native defaults already use None. Never allow native rmtree to remove
        # a caller's Artifact directory; checkpoint persistence is external.
        save_path = self.save_path
        self.save_path = None
        try:
            with torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled, benchmark=False,
                deterministic=True, allow_tf32=torch.backends.cudnn.allow_tf32,
            ):
                original_test(self, series)
        finally:
            self.save_path = save_path
        destination = getattr(self, "_fcvae_replay_destination", None)
        if destination is not None:
            save_replay(self, destination)

    module.OneByOneDataset, module.AllInOneDataset = curve_input, shared_input
    method.__init__, method.test_phase = initialize, test
    method._timesigma_label_free = True
    return package
